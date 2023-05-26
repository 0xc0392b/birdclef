from typing import List, Tuple, Dict, Iterator, Generator, Callable
from hashlib import sha1
from statistics import NormalDist
from csv import DictReader as CSVDictReader
from random import choices as random_choices
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from numpy import ndarray, array, matrix
from numpy import min as ndarray_min
from numpy import mean as ndarray_mean
from numpy import std as ndarray_std
from numpy import arange as ndarray_arange
from numpy import vectorize as vectorize_func
from scipy.ndimage import label as ndarray_label_features
from scipy.ndimage import maximum_position as ndarray_extract_region_maximums

from .dataset import Dataset


class CandidatePeak:
    def __init__(self, x: int, y: int, frequency: int) -> None:
        self._x = x
        self._y = y
        self._frequency = frequency

    def __str__(self) -> str:
        return f"({self._x}, {self._y}), {self._frequency}hz"

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def time(self) -> int:
        return self._x

    @property
    def frequency(self) -> int:
        return self._frequency


class Fingerprint:
    HASH_FUNCTION = sha1

    def __init__(self, hash_value: int, offset: int, label: str) -> None:
        self._hash_value = hash_value
        self._offset = offset
        self._label = label

    def __str__(self) -> str:
        return f"{self._label} {self._offset} {self._hash_value.hexdigest()}"

    @property
    def hash_value(self) -> int:
        return self._hash_value

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def label(self) -> str:
        return self._label

    @classmethod
    def from_pair(
            cls,
            a: CandidatePeak,
            b: CandidatePeak,
            hash_func: Callable,
            offset: int,
            label: str
    ) -> "Fingerprint":
        delta = abs(a.time - b.time)
        f1, f2 = a.frequency, b.frequency
        hash_value = hash_func(f"{f1}|{f2}|{delta}".encode("UTF-8"))
        return cls(hash_value, offset, label)


class ConstellationMap:
    def __init__(self, candidate_peaks: List[CandidatePeak]) -> None:
        self._candidate_peaks = candidate_peaks

    def __len__(self) -> int:
        return len(self._candidate_peaks)

    def __iter__(self) -> Iterator[CandidatePeak]:
        return iter(self._candidate_peaks)

    def to_vectors(self) -> Tuple[ndarray, ndarray, ndarray]:
        xs = array([p.x for p in self])
        ys = array([p.y for p in self])
        fs = array([p.frequency for p in self])
        return (xs, ys, fs)

    def fingerprints(
            self,
            label: str,
            region_size: int,
            hash_func: Callable
    ) -> Generator[Fingerprint, None, None]:
        for anchor_point in self:
            start = anchor_point.time
            end = start + region_size
            region = filter(lambda x: start < x.time <= end, self)

            for target_point in region:
                yield Fingerprint.from_pair(
                    a=anchor_point,
                    b=target_point,
                    hash_func=hash_func,
                    offset=start,
                    label=label
                )

    @classmethod
    def from_spectrogram(
            cls,
            spectrogram: ndarray,
            threshold: float
    ) -> "ConstellationMap":
        flattened = matrix.flatten(spectrogram)
        filtered = flattened[flattened > ndarray_min(flattened)]

        ndist = NormalDist(ndarray_mean(filtered), ndarray_std(filtered))
        zscore = vectorize_func(lambda x: ndist.zscore(x))
        zscore_matrix = zscore(spectrogram)

        mask_matrix = zscore_matrix > threshold
        labelled_matrix, num_regions = ndarray_label_features(mask_matrix)
        label_indices = ndarray_arange(num_regions) + 1

        peak_positions = ndarray_extract_region_maximums(
            zscore_matrix, labelled_matrix, label_indices)

        return cls([
            CandidatePeak(
                x=x,
                y=y,
                frequency=spectrogram[y, x]
            )
            for y, x in peak_positions
        ])


class HashTable:
    PATH = "/media/william/Scratch/output/birdclef-2023/hashtable"

    def __init__(self, dictionary: Dict[int, int]) -> None:
        self._dictionary = dictionary

    def __len__(self) -> int:
        return len(self._dictionary)

    def __getitem__(self, key: int) -> List[Tuple[str, int]]:
        if key in self._dictionary:
            return self._dictionary[key]
        else:
            return None

    def save_to_disk(self, hashtable_path: str) -> None:
        with open(f"{hashtable_path}/dictionary.pkl", "wb") as outfile:
            pickle_dump(self._dictionary, outfile)

    @classmethod
    def from_file(cls, hashtable_path: str) -> "HashTable":
        with open(f"{hashtable_path}/dictionary.pkl", "rb") as infile:
            dictionary = pickle_load(infile)
            return HashTable(dictionary)

    @classmethod
    def from_dataset(
            cls,
            dataset: Dataset,
            fingerprint_path: str,
            pick: int,
            mask: int
    ) -> "HashTable":
        dictionary = {}

        for label in dataset.labels():
            samples = dataset.with_label(label)
            sample_hashes = []

            for sample in samples:
                path = f"{fingerprint_path}/{sample.audio_file_name}.csv"

                with open(path, "rt") as csv:
                    for row in CSVDictReader(csv):
                        key = int(row["hash"], 16) & mask
                        value = int(row["offset"])
                        sample_hashes.append((key, value))

            for hash_value, offset in set(random_choices(sample_hashes, k=pick)):
                if hash_value not in dictionary:
                    dictionary[hash_value] = [(label, offset)]
                else:
                    dictionary[hash_value].append((label, offset))

            print(f"loaded {label}")

        return HashTable(dictionary)
