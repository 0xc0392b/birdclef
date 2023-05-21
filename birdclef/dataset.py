from typing import List, Tuple, Set, Dict, Iterator
from random import choice as random_choice
from csv import DictReader as CSVDictReader

from numpy import ndarray
from librosa import load as load_audio


class Sample:
    def __init__(
            self,
            label: str,
            author: str,
            dataset_path: str,
            audio_directory: str,
            audio_file_path: str,
            longitude: float,
            latitude: float
    ) -> None:
        self._label = label
        self._author = author
        self._dataset_path = dataset_path
        self._audio_directory = audio_directory
        self._audio_file_path = audio_file_path
        self._longitude = longitude
        self._latitude = latitude

    @property
    def label(self) -> str:
        return self._label

    @property
    def author(self) -> str:
        return self._author

    @property
    def audio_file_name(self) -> str:
        return self._audio_file_path

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self._longitude, self._latitude)

    @property
    def audio_file_path(self) -> str:
        return f"{self._dataset_path}/{self._audio_directory}/{self._audio_file_path}"

    def audio_samples(self, desired_sample_rate: int) -> ndarray:
        samples, _actual_sample_rate = load_audio(
            path=self.audio_file_path,
            sr=desired_sample_rate
        )

        return samples


class Dataset:
    PATH = "/media/william/Scratch/datasets/birdclef-2023"
    METADATA_FILE = "train_metadata.csv"
    AUDIO_DIRECTORY = "train_audio"
    SAMPLE_RATE = 32000

    def __init__(self, samples: List[Sample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def labels(self) -> Set[str]:
        return set([sample.label for sample in self])

    def with_label(self, label: str) -> List[Sample]:
        return list(filter(lambda x: x.label == label, self))

    def pick_random(self) -> Sample:
        return random_choice(self._samples)

    def sample(self, n: int) -> "Dataset":
        return Dataset([self.pick_random() for _ in range(n)])

    @classmethod
    def load(cls, path: str) -> "Dataset":
        with open(f"{path}/{Dataset.METADATA_FILE}", "r") as csv:
            return cls([
                Sample(
                    dataset_path=path,
                    audio_directory=Dataset.AUDIO_DIRECTORY,
                    audio_file_path=row["filename"],
                    label=row["primary_label"],
                    author=row["author"],
                    longitude=row["longitude"],
                    latitude=row["latitude"]
                )
                for row in CSVDictReader(csv)
            ])


class SummaryStatistics:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def num_samples(self) -> int:
        return len(self._dataset)

    def label_counts(self) -> Dict[str, int]:
        counts = {}

        for sample in self._dataset:
            if sample.label not in counts:
                counts[sample.label] = 1
            else:
                counts[sample.label] += 1
        
        return counts

    def audio_sample_counts(self) -> List[int]:
        return [
            len(sample.audio_samples(Dataset.SAMPLE_RATE))
            for sample in self._dataset
        ]
