import argparse
from pathlib import Path
import unittest

import pandas as pd

import d1_solver as s


DATA_DIR = None
RESULT_DIR = None


class TestQ1Solution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.center, cls.services = s.read_nodes(DATA_DIR / "调度中心与服务区.xlsx")
        cls.models = s.read_models(DATA_DIR / "运输无人机数据.xlsx")
        cls.boxes = s.read_boxes(DATA_DIR / "物资需求与配送时限.xlsx")
        cls.dem = s.GeoTiffDem(DATA_DIR / "镇龙乡及周边30米DEM.tif")
        cls.geometries = {sid: s.route_geometry(cls.center, node, cls.dem) for sid, node in cls.services.items()}
        cls.batches = pd.read_csv(RESULT_DIR / "Q1_单点组批.csv")

    def test_every_box_exactly_once(self):
        assigned = []
        for value in self.batches["货箱编号列表"]:
            assigned.extend(str(value).split(";"))
        expected = self.boxes["货箱编号"].astype(str).tolist()
        self.assertEqual(sorted(assigned), sorted(expected))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_every_batch_is_feasible(self):
        box_index = self.boxes.set_index("货箱编号")
        for _, row in self.batches.iterrows():
            ids = str(row["货箱编号列表"]).split(";")
            selected = box_index.loc[ids]
            sid = str(row["服务区编号"])
            self.assertTrue((selected["服务区编号"].astype(str) == sid).all())
            mass = float(selected["单箱质量（kg）"].sum())
            volume = float(selected["单箱体积（m³）"].sum())
            model = self.models[str(row["机型编号"])]
            energy = s.round_trip_energy_kwh(model, self.geometries[sid], mass)
            self.assertLessEqual(mass, model.max_payload_kg + 1e-8)
            self.assertLessEqual(volume, model.volume_m3 + 1e-10)
            self.assertLessEqual(energy, (1 - model.reserve) * model.battery_kwh + 1e-8)
            self.assertAlmostEqual(mass, float(row["总质量（kg）"]), places=6)
            self.assertAlmostEqual(volume, float(row["总体积（m³）"]), places=6)
            self.assertAlmostEqual(energy, float(row["架次能耗（kWh）"]), places=6)

    def test_payload_limit_is_monotone_and_safe(self):
        for geom in self.geometries.values():
            for model in self.models.values():
                q = s.maximum_safe_payload(model, geom, model.reserve)
                e = s.round_trip_energy_kwh(model, geom, q)
                self.assertLessEqual(e, (1 - model.reserve) * model.battery_kwh + 1e-8)
                if q < model.max_payload_kg - 1e-5:
                    e_above = s.round_trip_energy_kwh(model, geom, min(model.max_payload_kg, q + 1e-4))
                    self.assertGreater(e_above, (1 - model.reserve) * model.battery_kwh - 1e-7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    DATA_DIR = args.data_dir
    RESULT_DIR = args.result_dir
    unittest.main(argv=[__file__, *remaining])
