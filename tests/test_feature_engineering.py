from unittest import TestCase

from birdclef import Pipeline


class TestPipeline(TestCase):
    def test_build_pipeline_1(self):
        pipe = Pipeline.build_pipeline_1()
        self.assertEqual(len(pipe), 2)

    def test_build_pipeline_2(self):
        pipe = Pipeline.build_pipeline_2()
        self.assertEqual(len(pipe), 5)

    def test_build_pipeline_3(self):
        pipe = Pipeline.build_pipeline_2()
        self.assertEqual(len(pipe), 6)

    def test_call(self):
        pass
