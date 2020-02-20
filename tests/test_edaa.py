import unittest
from DefectDetection.datasets.steel_data import load_img

class TestEDAA(unittest.TestCase):
    def test_import(self):
        defect_names, defect_img = load_img()
