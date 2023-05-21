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
            output_path=SpectrogramPreprocessor.PATH,
            feature_pipeline=pipeline
        )

        processor.run(
            num_workers=SpectrogramPreprocessor.NUM_WORKERS,
            dataset=dataset
        )


class TestFingerprintPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH).sample(10)

        processor = FingerprintPreprocessor(
            input_path=SpectrogramPreprocessor.PATH,
            output_path=FingerprintPreprocessor.PATH
        )

        processor.run(
            num_workers=FingerprintPreprocessor.NUM_WORKERS,
            region_size=100,
            threshold=2.75,
            dataset=dataset
        )
