from typing import List

from numpy import ndarray

from .dataset import Dataset
from .soundscape import Soundscape


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
    def __init__(self, labels: List[Label]) -> None:
        self._labels = label

    def __call__(self, soundscape: Soundscape) -> ResultSet:

        # TODO determine which birds were in the soundscape
        # and wrap the classifications in a result set object
        # ...

        return


class Evaluator:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def evaluate(self, classifier: Classifier) -> Something:

        # TODO measure the classifier's performance
        # ...

        return
