from unittest import TestCase

from birdclef import Dataset
from birdclef import FeaturePipeline
from birdclef import SpectrogramPreprocessor, FingerprintPreprocessor, PeakPreprocessor


class TestSpectrogramPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH)
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


class TestPeakPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH)

        processor = PeakPreprocessor(
            input_path=SpectrogramPreprocessor.PATH,
            output_path=PeakPreprocessor.PATH
        )

        processor.run(
            num_workers=PeakPreprocessor.NUM_WORKERS,
            threshold=2.75,
            dataset=dataset
        )


class TestFingerprintPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH)

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
