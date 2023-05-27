from typing import List, Tuple, Dict, Iterator, Generator, Callable
from hashlib import sha1
from statistics import NormalDist

from numpy import ndarray, array, matrix
from numpy import min as ndarray_min
from numpy import mean as ndarray_mean
from numpy import std as ndarray_std
from numpy import arange as ndarray_arange
from numpy import vectorize as vectorize_func
from scipy.ndimage import label as ndarray_label_features
from scipy.ndimage import maximum_position as ndarray_extract_region_maximums


class CandidatePeak:
    def __init__(self, time: int, freq: int, db: int) -> None:
        self._time = time
        self._freq = freq
        self._db = db

    def __str__(self) -> str:
        return f"{self._time} | {self._freq}hz | {self._db}db"

    @property
    def time(self) -> int:
        return self._time

    @property
    def freq(self) -> int:
        return self._freq

    @property
    def db(self) -> int:
        return self._db


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
        f1, f2 = a.freq, b.freq
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
        time = array([p.time for p in self])
        freq = array([p.freq for p in self])
        db = array([p.db for p in self])
        return (time, freq, db)

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
                time=x,
                freq=y,
                db=spectrogram[y, x]
            )
            for y, x in peak_positions
        ])
