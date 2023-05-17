from typing import Any, Callable, List
from functools import reduce

from numpy import ndarray
from librosa import stft, amplitude_to_db
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Stage:
    def __call__(self, x: ndarray) -> ndarray:
        raise NotImplementedError


class Transpose(Stage):
    def __call__(self, x: ndarray) -> ndarray:
        return x.T


class ShortTimeFourierTransform(Stage):
    def __call__(self, x: ndarray) -> ndarray:
        return stft(x)


class AmplitudeToDBSpectrogram(Stage):
    def __call__(self, x: ndarray) -> ndarray:
        return amplitude_to_db(abs(x))


class Standardisation(Stage):
    def __call__(self, x: ndarray) -> ndarray:
        return StandardScaler().fit_transform(x)


class PrincipalComponentAnalysis(Stage):
    def __init__(self, n_components: int) -> None:
        self._n_components = n_components

    def __call__(self, x: ndarray) -> ndarray:
        return PCA(n_components=self._n_components).fit_transform(x)


class Pipeline:
    def __init__(self, stages: List[Stage]) -> None:
        self._stages = stages

    @staticmethod
    def foldl(func: Callable, acc: Any, xs: List[Any]) -> Any:
        return reduce(func, xs, acc)

    @staticmethod
    def do_stage(x: ndarray, stage: Callable) -> ndarray:
        return stage(x)

    def __len__(self) -> int:
        return len(self._stages)

    def __call__(self, x: ndarray) -> ndarray:
        return Pipeline.foldl(Pipeline.do_stage, x, self._stages)

    @classmethod
    def build_pipeline_1(cls) -> "Pipeline":
        """
        fourier transform
        spectrogram
        """
        return cls([
            ShortTimeFourierTransform(),
            AmplitudeToDBSpectrogram()
        ])

    @classmethod
    def build_pipeline_2(cls, n_components: int) -> "Pipeline":
        """
        fourier transform
        spectrogram
        dimensionality reduction (PCA)
        """
        return cls([
            ShortTimeFourierTransform(),
            AmplitudeToDBSpectrogram(),
            Transpose(),
            PrincipalComponentAnalysis(n_components),
            Transpose()
        ])

    @classmethod
    def build_pipeline_3(cls, n_components: int) -> "Pipeline":
        """
        fourier transform
        spectrogram
        standardise
        dimensionality reduction (PCA)
        """
        return cls([
            ShortTimeFourierTransform(),
            AmplitudeToDBSpectrogram(),
            Transpose(),
            Standardisation(),
            PrincipalComponentAnalysis(n_components),
            Transpose()
        ])
