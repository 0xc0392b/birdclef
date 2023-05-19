from unittest import TestCase

from birdclef import Dataset
from birdclef import FeaturePipeline
from birdclef import SpectrogramPreprocessor, FingerprintPreprocessor


class TestSpectrogramPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH).sample(10)
        pipeline = FeaturePipeline.build_pipeline_1()

        processor = SpectrogramPreprocessor(
            sample_rate=Dataset.SAMPLE_RATE,
            output_path=Preprocessor.PATH,
            feature_pipeline=pipeline
        )

        processor.run(Preprocessor.NUM_WORKERS, dataset)


class TestFingerprintPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH).sample(10)

        processor = FingerprintPreprocessor(
            sample_rate=Dataset.SAMPLE_RATE,
            output_path=Preprocessor.PATH
        )

        processor.run(Preprocessor.NUM_WORKERS, dataset)
