from unittest import TestCase

from birdclef import FeaturePipeline


class TestFeaturePipeline(TestCase):
    def test_build_pipeline_1(self):
        pipe = FeaturePipeline.build_pipeline_1()
        self.assertEqual(len(pipe), 2)

    def test_build_pipeline_2(self):
        pipe = FeaturePipeline.build_pipeline_2(n_components=10)
        self.assertEqual(len(pipe), 5)

    def test_build_pipeline_3(self):
        pipe = FeaturePipeline.build_pipeline_3(n_components=10)
        self.assertEqual(len(pipe), 6)
