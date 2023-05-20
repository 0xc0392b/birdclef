from typing import List

from numpy import ndarray

from .dataset import Dataset
from .soundscape import Soundscape
from .feature_engineering import FeaturePipeline


class Classification:
    def __init__(self, timestamp: int, label: str) -> None:
        self._timestamp = timestamp
        self._label = label

    def __str__(self) -> str:
        return "something"

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def label(self) -> str:
        return self._label


class ResultSet:
    def __init__(
            self,
            soundscape: Soundscape,
            labels: List[str],
            classifications: List[Classification]
    ) -> None:
        self._soundscape = soundscape
        self._labels = labels
        self._classifications = classifications

    def __str__(self) -> str:
        return "something"

    @property
    def classifications(self) -> List[Classification]:
        return self._classifications

    def to_vector(self) -> ndarray:
        return

    def save_to_file(self, path: str) -> None:
        return


class Classifier:
    def __init__(self, labels: List[str]) -> None:
        self._labels = label

    def __call__(self, soundscape: Soundscape) -> ResultSet:

        # TODO determine which birds were in the soundscape
        # and wrap the classifications in a result set object
        # ...

        return


class ModelEvaluation:
    def __init__(self, accuracy: float) -> None:
        self._accuracy = accuracy

    def __str__(self) -> str:
        return f"{self._accuracy * 100}% accuracy"

    @property
    def accuracy(self) -> float:
        return self._accuracy


class Evaluator:
    def __init__(
            self,
            dataset: Dataset,
            feature_pipeline: FeaturePipeline
    ) -> None:
        self._dataset = dataset
        self._feature_pipeline = feature_pipeline

    def evaluate(self, classifier: Classifier) -> ModelEvaluation:

        # TODO measure the classifier's performance
        # ...

        return
