from unittest import TestCase

from birdclef import Dataset
from birdclef import FeaturePipeline
from birdclef import SpectrogramPreprocessor, FingerprintPreprocessor
from birdclef import Classifier, ResultSet, Evaluator


class TestClassifier(UnitTest):
    def test_classify(self):
        dataset = Dataset.load(Dataset.PATH).sample(10)
        pipeline = FeaturePipeline.build_pipeline_1()

        processor_1 = SpectrogramPreprocessor(
            sample_rate=Dataset.SAMPLE_RATE,
            output_path=Preprocessor.PATH,
            feature_pipeline=pipeline
        )

        processor_2 = FingerprintPreprocessor(
            sample_rate=Dataset.SAMPLE_RATE,
            output_path=Preprocessor.PATH
        )

        classifier = Classifier(model=Something())


class TestResultSet(UnitTest):
    def test_to_vector(self):
        pass

    def test_save_to_file(self):
        pass


class TestEvaluator(UnitTest):
    def test_evaluate(self):
        pass
