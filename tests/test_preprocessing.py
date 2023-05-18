from unittest import TestCase

from birdclef import Dataset, FeaturePipeline, Preprocessor


class TestPreprocessor(TestCase):
    def test_run(self):
        dataset = Dataset.load(Dataset.PATH).sample(10)
        processor = Preprocessor(
            sample_rate=Dataset.SAMPLE_RATE,
            output_path=Preprocessor.PATH,
            feature_pipeline=FeaturePipeline.build_pipeline_1()
        )

        processor.run(Preprocessor.NUM_WORKERS, dataset)

