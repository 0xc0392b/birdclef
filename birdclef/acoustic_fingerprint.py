from typing import List, Tuple, Iterator, Generator, Callable
from hashlib import sha1

from numpy import array, ndarray


class CandidatePeak:
    def __init__(self, time: int, frequency: int) -> None:
        self._time = time
        self._frequency = frequency

    def __str__(self) -> str:
        return f"ts: {self._time}, hz: {self._frequency}"

    @property
    def time(self) -> int:
        return self._time

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

    def to_vectors(self) -> Tuple[ndarray, ndarray]:
        times = array([p.time for p in self])
        frequencies = array([p.frequency for p in self])
        return (times, frequencies)

    def fingerprints(self) -> Generator[Fingerprint]:
        for peak in self:
            yield Fingerprint()

    @classmethod
    def from_spectrogram(cls, spectrogram: ndarray) -> "ConstellationMap":

        flattened = np.matrix.flatten(spectrogram)
        filtered = flattened[flattened > np.min(flattened)]

        mean, std = np.mean(filtered), np.std(filtered)
        ndist = NormalDist(mu=mean, sigma=std)

        zscore = np.vectorize(lambda x: ndist.zscore(x))
        zscore_matrix = zscore(x1)

        labeled_matrix, num_features = ndimage.label(zscore_matrix > 2.75)
        max_values = ndimage.maximum_position(
            zscore_matrix, labeled_matrix, np.arange(num_features) + 1)

        return cls([
            CandidatePeak()
            for x, y in max_values
        ])
