from numpy import ndarray
from librosa import load as load_audio


class Soundscape:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    @property
    def file_path(self) -> str:
        return self._file_path

    def audio_samples(self, desired_sample_rate: int) -> ndarray:
        samples, _actual_sample_rate = load_audio(
            path=self.file_path,
            sr=desired_sample_rate
        )

        return samples
