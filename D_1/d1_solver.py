from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image


G = 9.80665
JOULE_PER_KWH = 3.6e6
EARTH_RADIUS_M = 6_371_008.8
DEFAULT_RESERVES = (0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class Node:
    node_id: str
    name: str
    lon: float
    lat: float
    elevation_m: float


@dataclass(frozen=True)
class DroneModel:
    model_id: str
    name: str
    empty_mass_kg: float
    max_payload_kg: float
    volume_m3: float
    cruise_speed_mps: float
    range_empty_m: float
    range_full_m: float
    battery_kwh: float
    reserve: float
    prep_s: float
    load_per_box_s: float
    handoff_base_s: float
    handoff_per_box_s: float
    climb_speed_mps: float
    descend_speed_mps: float
    climb_eff: float
    descend_eff: float


@dataclass(frozen=True)
class BoxKind:
    material: str
    mass_kg: float
    volume_m3: float
    ids: tuple[str, ...]


@dataclass(frozen=True)
class RouteGeometry:
    service_id: str
    horizontal_m: float
    max_dem_m: float
    cruise_altitude_m: float
    outbound_climb_m: float
    outbound_descend_m: float
    return_climb_m: float
    return_descend_m: float


@dataclass(frozen=True)
class BatchCandidate:
    counts: tuple[int, ...]
    model_id: str
    mass_kg: float
    volume_m3: float
    energy_kwh: float
    operation_time_s: float
    return_soc: float


@dataclass(frozen=True)
class PlanCost:
    sorties: int
    energy_kwh: float
    time_s: float

    def plus(self, batch: BatchCandidate) -> "PlanCost":
        return PlanCost(
            self.sorties + 1,
            self.energy_kwh + batch.energy_kwh,
            self.time_s + batch.operation_time_s,
        )


class GeoTiffDem:
    """Reads the supplied float GeoTIFF using Pillow and its GeoTIFF tags."""

    def __init__(self, path: Path):
        image = Image.open(path)
        self.data = np.asarray(image, dtype=float)
        tie = tuple(float(x) for x in image.tag_v2[33922])
        scale = tuple(float(x) for x in image.tag_v2[33550])
        self.lon0 = tie[3]
        self.lat0 = tie[4]
        self.dx = scale[0]
        self.dy = scale[1]
        self.nodata = -32767.0
        self.height, self.width = self.data.shape

    def fractional_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        col = (lon - self.lon0) / self.dx
        row = (self.lat0 - lat) / self.dy
        return col, row

    def value(self, row: int, col: int) -> float:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"DEM sample outside raster: row={row}, col={col}")
        value = float(self.data[row, col])
        if not math.isfinite(value) or value <= self.nodata + 1:
            raise ValueError(f"NoData on route: row={row}, col={col}")
        return value

    def traversed_cells(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> list[tuple[int, int]]:
        """Amanatides-Woo grid traversal in pixel-cell coordinates."""
        c0, r0 = self.fractional_pixel(lon1, lat1)
        c1, r1 = self.fractional_pixel(lon2, lat2)
        x0, y0 = c0 + 0.5, r0 + 0.5
        x1, y1 = c1 + 0.5, r1 + 0.5
        dx, dy = x1 - x0, y1 - y0
        ix, iy = math.floor(x0), math.floor(y0)
        end_x, end_y = math.floor(x1), math.floor(y1)
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        t_delta_x = math.inf if dx == 0 else abs(1.0 / dx)
        t_delta_y = math.inf if dy == 0 else abs(1.0 / dy)
        next_x = (ix + 1) if step_x > 0 else ix
        next_y = (iy + 1) if step_y > 0 else iy
        t_max_x = math.inf if dx == 0 else (next_x - x0) / dx
        t_max_y = math.inf if dy == 0 else (next_y - y0) / dy

        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while True:
            cell = (iy, ix)
            if cell not in seen:
                cells.append(cell)
                seen.add(cell)
            if ix == end_x and iy == end_y:
                break
            if t_max_x < t_max_y:
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_x:
                iy += step_y
                t_max_y += t_delta_y
            else:
                # At a corner the line touches the two side cells as well.
                side_x = (iy, ix + step_x)
                side_y = (iy + step_y, ix)
                for side in (side_x, side_y):
                    if side not in seen:
                        cells.append(side)
                        seen.add(side)
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
        return cells

    def maximum_on_segment(self, a: Node, b: Node) -> float:
        return max(self.value(row, col) for row, col in self.traversed_cells(a.lon, a.lat, b.lon, b.lat))


def haversine_m(a: Node, b: Node) -> float:
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = p2 - p1
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def read_nodes(path: Path) -> tuple[Node, dict[str, Node]]:
    raw = pd.read_excel(path, sheet_name="数据", header=None)
    center = Node(
        str(raw.iloc[2, 0]).strip(), str(raw.iloc[2, 1]).strip(),
        float(raw.iloc[2, 2]), float(raw.iloc[2, 3]), float(raw.iloc[2, 4]),
    )
    services: dict[str, Node] = {}
    for _, row in raw.iloc[6:].iterrows():
        if pd.isna(row.iloc[0]):
            continue
        node = Node(str(row.iloc[0]).strip(), str(row.iloc[1]).strip(), float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4]))
        services[node.node_id] = node
    return center, services


def read_models(path: Path) -> dict[str, DroneModel]:
    raw = pd.read_excel(path, sheet_name="数据", header=None)
    models: dict[str, DroneModel] = {}
    for _, row in raw.iloc[2:5].iterrows():
        values = row.iloc[:18].tolist()
        model = DroneModel(
            model_id=str(values[0]).strip(), name=str(values[1]).strip(),
            empty_mass_kg=float(values[2]), max_payload_kg=float(values[3]), volume_m3=float(values[4]),
            cruise_speed_mps=float(values[5]), range_empty_m=float(values[6]), range_full_m=float(values[7]),
            battery_kwh=float(values[8]), reserve=float(values[9]) / 100.0,
            prep_s=float(values[10]), load_per_box_s=float(values[11]),
            handoff_base_s=float(values[12]), handoff_per_box_s=float(values[13]),
            climb_speed_mps=float(values[14]), descend_speed_mps=float(values[15]),
            climb_eff=float(values[16]), descend_eff=float(values[17]),
        )
        models[model.model_id] = model
    return models


def read_boxes(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="逐箱货箱清单")
    df.columns = [str(c).strip() for c in df.columns]
    needed = ["货箱编号", "服务区编号", "物资类型", "单箱质量（kg）", "单箱体积（m³）"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing box columns: {missing}")
    return df


def route_geometry(center: Node, service: Node, dem: GeoTiffDem) -> RouteGeometry:
    max_dem = dem.maximum_on_segment(center, service)
    cruise = max_dem + 50.0
    base_work = center.elevation_m
    service_work = service.elevation_m + 30.0
    return RouteGeometry(
        service.node_id,
        haversine_m(center, service), max_dem, cruise,
        max(0.0, cruise - base_work), max(0.0, cruise - service_work),
        max(0.0, cruise - service_work), max(0.0, cruise - base_work),
    )


def equivalent_range_m(model: DroneModel, payload_kg: float) -> float:
    q = min(max(payload_kg, 0.0), model.max_payload_kg)
    ratio = q / model.max_payload_kg if model.max_payload_kg > 0 else 0.0
    return model.range_empty_m - (model.range_empty_m - model.range_full_m) * ratio ** 1.5


def segment_energy_kwh(model: DroneModel, distance_m: float, climb_m: float, payload_kg: float) -> float:
    horizontal = model.battery_kwh * distance_m / equivalent_range_m(model, payload_kg)
    climb = (model.empty_mass_kg + payload_kg) * G * climb_m / (model.climb_eff * JOULE_PER_KWH)
    return horizontal + climb


def round_trip_energy_kwh(model: DroneModel, geom: RouteGeometry, payload_kg: float) -> float:
    outbound = segment_energy_kwh(model, geom.horizontal_m, geom.outbound_climb_m, payload_kg)
    inbound = segment_energy_kwh(model, geom.horizontal_m, geom.return_climb_m, 0.0)
    return outbound + inbound


def round_trip_flight_time_s(model: DroneModel, geom: RouteGeometry) -> float:
    out = geom.outbound_climb_m / model.climb_speed_mps + geom.horizontal_m / model.cruise_speed_mps + geom.outbound_descend_m / model.descend_speed_mps
    back = geom.return_climb_m / model.climb_speed_mps + geom.horizontal_m / model.cruise_speed_mps + geom.return_descend_m / model.descend_speed_mps
    return out + back


def operation_time_s(model: DroneModel, geom: RouteGeometry, box_count: int) -> float:
    return (
        model.prep_s + box_count * model.load_per_box_s
        + round_trip_flight_time_s(model, geom)
        + model.handoff_base_s + box_count * model.handoff_per_box_s
    )


def maximum_safe_payload(model: DroneModel, geom: RouteGeometry, reserve: float) -> float:
    limit = (1.0 - reserve) * model.battery_kwh
    if round_trip_energy_kwh(model, geom, 0.0) > limit + 1e-12:
        return 0.0
    if round_trip_energy_kwh(model, geom, model.max_payload_kg) <= limit + 1e-12:
        return model.max_payload_kg
    lo, hi = 0.0, model.max_payload_kg
    for _ in range(70):
        mid = (lo + hi) / 2.0
        if round_trip_energy_kwh(model, geom, mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def make_box_kinds(service_boxes: pd.DataFrame) -> list[BoxKind]:
    kinds: list[BoxKind] = []
    grouped = service_boxes.groupby(["物资类型", "单箱质量（kg）", "单箱体积（m³）"], sort=True)
    for (material, mass, volume), group in grouped:
        ids = tuple(sorted(group["货箱编号"].astype(str).tolist()))
        kinds.append(BoxKind(str(material), float(mass), float(volume), ids))
    return kinds


def all_batch_candidates(
    kinds: Sequence[BoxKind], models: dict[str, DroneModel], geom: RouteGeometry, reserve: float
) -> list[BatchCandidate]:
    max_counts = [len(k.ids) for k in kinds]
    payload_limits = {m.model_id: maximum_safe_payload(m, geom, reserve) for m in models.values()}
    result: list[BatchCandidate] = []
    for counts in product(*(range(n + 1) for n in max_counts)):
        box_count = sum(counts)
        if box_count == 0:
            continue
        mass = sum(c * k.mass_kg for c, k in zip(counts, kinds))
        volume = sum(c * k.volume_m3 for c, k in zip(counts, kinds))
        for model in models.values():
            if mass > min(model.max_payload_kg, payload_limits[model.model_id]) + 1e-9:
                continue
            if volume > model.volume_m3 + 1e-12:
                continue
            energy = round_trip_energy_kwh(model, geom, mass)
            limit = (1.0 - reserve) * model.battery_kwh
            if energy > limit + 1e-9:
                continue
            result.append(BatchCandidate(
                tuple(counts), model.model_id, mass, volume, energy,
                operation_time_s(model, geom, box_count),
                1.0 - energy / model.battery_kwh,
            ))
    return result


def objective_key(cost: PlanCost, mode: str) -> tuple[float, float, float]:
    if mode == "energy_first":
        return cost.sorties, cost.energy_kwh, cost.time_s
    if mode == "time_first":
        return cost.sorties, cost.time_s, cost.energy_kwh
    raise ValueError(f"Unknown objective mode: {mode}")


def solve_service(
    kinds: Sequence[BoxKind], candidates: Sequence[BatchCandidate], mode: str
) -> tuple[PlanCost, list[BatchCandidate]]:
    full_state = tuple(len(k.ids) for k in kinds)
    candidates = tuple(candidates)

    @lru_cache(maxsize=None)
    def dp(state: tuple[int, ...]) -> tuple[PlanCost, tuple[int, ...]] | None:
        if not any(state):
            return PlanCost(0, 0.0, 0.0), ()
        pivot = next(i for i, n in enumerate(state) if n > 0)
        best: tuple[PlanCost, tuple[int, ...]] | None = None
        for idx, batch in enumerate(candidates):
            if batch.counts[pivot] == 0:
                continue
            if any(c > s for c, s in zip(batch.counts, state)):
                continue
            next_state = tuple(s - c for s, c in zip(state, batch.counts))
            tail = dp(next_state)
            if tail is None:
                continue
            cost = tail[0].plus(batch)
            plan = (idx,) + tail[1]
            if best is None or objective_key(cost, mode) < objective_key(best[0], mode):
                best = cost, plan
        return best

    solved = dp(full_state)
    if solved is None:
        raise RuntimeError(f"No feasible batching plan for counts={full_state}")
    return solved[0], [candidates[i] for i in solved[1]]


def assign_box_ids(kinds: Sequence[BoxKind], plan: Sequence[BatchCandidate]) -> list[list[str]]:
    remaining = [list(k.ids) for k in kinds]
    assigned: list[list[str]] = []
    for batch in plan:
        ids: list[str] = []
        for i, count in enumerate(batch.counts):
            ids.extend(remaining[i][:count])
            del remaining[i][:count]
        assigned.append(ids)
    if any(remaining):
        raise AssertionError("Unassigned boxes remain")
    return assigned


def solve_all(data_dir: Path, output_dir: Path, reserves: Sequence[float]) -> None:
    center, services = read_nodes(data_dir / "调度中心与服务区.xlsx")
    models = read_models(data_dir / "运输无人机数据.xlsx")
    boxes = read_boxes(data_dir / "物资需求与配送时限.xlsx")
    dem = GeoTiffDem(data_dir / "镇龙乡及周边30米DEM.tif")
    geometries = {sid: route_geometry(center, service, dem) for sid, service in services.items()}

    payload_rows: list[dict] = []
    for sid, geom in geometries.items():
        for model in models.values():
            payload_rows.append({
                "服务区编号": sid,
                "机型编号": model.model_id,
                "水平单程距离（m）": geom.horizontal_m,
                "航段最高DEM（m）": geom.max_dem_m,
                "计划巡航海拔（m）": geom.cruise_altitude_m,
                "去程爬升（m）": geom.outbound_climb_m,
                "返程爬升（m）": geom.return_climb_m,
                "返航安全余量": model.reserve,
                "最大安全载荷（kg）": maximum_safe_payload(model, geom, model.reserve),
                "额定载荷（kg）": model.max_payload_kg,
            })

    batch_rows: list[dict] = []
    tradeoff_rows: list[dict] = []
    sortie_number = 1
    all_box_ids: list[str] = []
    for sid in sorted(services):
        service_boxes = boxes[boxes["服务区编号"].astype(str) == sid].copy()
        kinds = make_box_kinds(service_boxes)
        geom = geometries[sid]
        default_reserve = next(iter(models.values())).reserve
        candidates = all_batch_candidates(kinds, models, geom, default_reserve)
        energy_cost, energy_plan = solve_service(kinds, candidates, "energy_first")
        time_cost, time_plan = solve_service(kinds, candidates, "time_first")
        tradeoff_rows.extend([
            {"服务区编号": sid, "优先规则": "架次-能耗-时间", "架次数": energy_cost.sorties, "总能耗（kWh）": energy_cost.energy_kwh, "累计作业时间（s）": energy_cost.time_s},
            {"服务区编号": sid, "优先规则": "架次-时间-能耗", "架次数": time_cost.sorties, "总能耗（kWh）": time_cost.energy_kwh, "累计作业时间（s）": time_cost.time_s},
        ])
        assigned = assign_box_ids(kinds, energy_plan)
        for batch, ids in zip(energy_plan, assigned):
            batch_rows.append({
                "架次编号": f"Q1-{sortie_number:03d}",
                "服务区编号": sid,
                "机型编号": batch.model_id,
                "货箱编号列表": ";".join(ids),
                "总质量（kg）": batch.mass_kg,
                "总体积（m³）": batch.volume_m3,
                "往返时间（s）": batch.operation_time_s,
                "架次能耗（kWh）": batch.energy_kwh,
                "返航SOC（%）": 100.0 * batch.return_soc,
            })
            all_box_ids.extend(ids)
            sortie_number += 1

    if sorted(all_box_ids) != sorted(boxes["货箱编号"].astype(str).tolist()):
        raise AssertionError("Box coverage check failed")

    sensitivity_rows: list[dict] = []
    payload_sensitivity_rows: list[dict] = []
    for reserve in reserves:
        for sid, geom in geometries.items():
            payload_sensitivity_rows.append({
                "返航安全余量": reserve,
                "服务区编号": sid,
                **{f"{model.model_id}型最大安全载荷（kg）": maximum_safe_payload(model, geom, reserve) for model in models.values()},
            })
        total = PlanCost(0, 0.0, 0.0)
        model_counts = {mid: 0 for mid in models}
        for sid in sorted(services):
            service_boxes = boxes[boxes["服务区编号"].astype(str) == sid].copy()
            kinds = make_box_kinds(service_boxes)
            candidates = all_batch_candidates(kinds, models, geometries[sid], reserve)
            cost, plan = solve_service(kinds, candidates, "energy_first")
            total = PlanCost(total.sorties + cost.sorties, total.energy_kwh + cost.energy_kwh, total.time_s + cost.time_s)
            for batch in plan:
                model_counts[batch.model_id] += 1
        sensitivity_rows.append({
            "返航安全余量": reserve,
            "总架次数": total.sorties,
            "总能耗（kWh）": total.energy_kwh,
            "累计作业时间（s）": total.time_s,
            **{f"{mid}型架次数": count for mid, count in model_counts.items()},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload_rows).to_csv(output_dir / "最大安全载荷.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(batch_rows).to_csv(output_dir / "Q1_单点组批.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(tradeoff_rows).to_csv(output_dir / "目标权衡对比.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "返航余量敏感性.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(payload_sensitivity_rows).to_csv(output_dir / "最大安全载荷敏感性.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    summary = {
        "box_count": int(len(boxes)),
        "service_count": len(services),
        "sortie_count": len(batch_rows),
        "total_energy_kwh": float(sum(r["架次能耗（kWh）"] for r in batch_rows)),
        "total_operation_time_s": float(sum(r["往返时间（s）"] for r in batch_rows)),
        "model_sorties": pd.Series([r["机型编号"] for r in batch_rows]).value_counts().sort_index().to_dict(),
    }
    (output_dir / "汇总指标.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D题第一问：最大安全载荷与单点组批精确求解")
    parser.add_argument("--data-dir", type=Path, required=True, help="包含官方Excel与DEM.tif的目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="结果CSV/JSON输出目录")
    parser.add_argument("--reserves", type=float, nargs="*", default=list(DEFAULT_RESERVES), help="敏感性分析返航余量，如 0.1 0.2 0.3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve_all(args.data_dir, args.output_dir, args.reserves)
