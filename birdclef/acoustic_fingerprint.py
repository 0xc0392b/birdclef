from typing import List, Tuple, Iterator

from numpy import ndarray


class CandidatePeak:
    def __init__(self, time: int, frequency: int) -> None:
        self._time = time
        self._frequency = frequency

    def __str__(self) -> str:
        return "something"

    @property
    def time(self) -> int:
        return self._time

    @property
    def frequency(self) -> int:
        return self._frequency


class ConstellationMap:
    def __init__(self, candidate_peaks: List[CandidatePeak]) -> None:
        self._candidate_peaks = candidate_peaks

    def __len__(self) -> int:
        return len(self._candidate_peaks)

    def __iter__(self) -> Iterator[CandidatePeak]:
        return iter(self._candidate_peaks)

    def to_vectors(self) -> Tuple[ndarray, ndarray]:
        times = []
        frequencies = []
        return (times, frequencies)

    @classmethod
    def from_spectrogram(cls, spectrogram: ndarray) -> "ConstellationMap":

        # TODO peak detection algorithm
        # ...
        
        return cls([
            # ...
        ])


class Fingerprint:
    def __init__(self, hash_value: int, offset: int, label: str) -> None:
        self._hash_value = hash_value
        self._offset = offset
        self._label = label

    def __str__(self) -> str:
        return "something"

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
    def from_pair(cls, a: CandidatePeak, b: CandidatePeak) -> "Fingerprint":

        # TODO the hashing stuff
        # ...

        return Fingerprint()
