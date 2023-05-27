from os import makedirs as make_directory
from os.path import dirname as directory_name
from multiprocessing import Pool as WorkerPool
from csv import DictWriter as CSVDictWriter
from functools import partial

from numpy import save as ndarray_write_to_disk
from numpy import load as ndarray_load_from_disk

from .dataset import Dataset, Sample
from .acoustic_fingerprint import ConstellationMap, Fingerprint
from .feature_engineering import FeaturePipeline


class SpectrogramPreprocessor:
    NUM_WORKERS = 12
    PATH = "/media/william/Scratch/output/birdclef-2023/spectrograms"

    def __init__(
            self,
            sample_rate: int,
            output_path: str,
            feature_pipeline: FeaturePipeline
    ) -> None:
        self._sample_rate = sample_rate
        self._output_path = output_path
        self._feature_pipeline = feature_pipeline

    @staticmethod
    def do_work(
            sample: Sample,
            feature_pipeline: FeaturePipeline,
            sample_rate: int,
            output_path: str
    ) -> None:
        audio_samples = sample.audio_samples(sample_rate)
        features = feature_pipeline(audio_samples)
        path = f"{output_path}/{sample.audio_file_name}.npy"

        make_directory(directory_name(path), exist_ok=True)

        with open(path, "wb") as outfile:
            ndarray_write_to_disk(outfile, features)

        print(f"DONE {path}")

    def run(self, num_workers: int, dataset: Dataset) -> None:
        with WorkerPool(num_workers) as pool:
            pool.map(
                partial(
                    SpectrogramPreprocessor.do_work,
                    feature_pipeline=self._feature_pipeline,
                    sample_rate=self._sample_rate,
                    output_path=self._output_path
                ),
                dataset
            )


class PeakPreprocessor:
    NUM_WORKERS = 6
    PATH = "/media/william/Scratch/output/birdclef-2023/peaks"

    def __init__(
            self,
            input_path: str,
            output_path: str
    ) -> None:
        self._input_path = input_path
        self._output_path = output_path

    @staticmethod
    def do_work(
            sample: Sample,
            threshold: float,
            input_path: str,
            output_path: str
    ) -> None:
        npy_path = f"{input_path}/{sample.audio_file_name}.npy"
        csv_path = f"{output_path}/{sample.audio_file_name}.csv"

        make_directory(directory_name(csv_path), exist_ok=True)

        with open(npy_path, "rb") as infile:
            spectrogram = ndarray_load_from_disk(infile)

            cmap = ConstellationMap.from_spectrogram(
                spectrogram=spectrogram,
                threshold=threshold
            )

            with open(csv_path, "wt") as outfile:
                writer = CSVDictWriter(outfile, ["time", "freq", "db"])
                writer.writeheader()

                for peak in cmap:
                    writer.writerow({
                        "time": peak.time,
                        "freq": peak.freq,
                        "db": peak.db
                    })

        print(f"DONE {csv_path}")

    def run(
            self,
            num_workers: int,
            threshold: float,
            dataset: Dataset
    ) -> None:
        with WorkerPool(num_workers) as pool:
            pool.map(
                partial(
                    PeakPreprocessor.do_work,
                    threshold=threshold,
                    input_path=self._input_path,
                    output_path=self._output_path
                ),
                dataset
            )


class FingerprintPreprocessor:
    NUM_WORKERS = 12
    PATH = "/media/william/Scratch/output/birdclef-2023/fingerprints"

    def __init__(
            self,
            input_path: str,
            output_path: str
    ) -> None:
        self._input_path = input_path
        self._output_path = output_path

    @staticmethod
    def do_work(
            sample: Sample,
            threshold: float,
            region_size: int,
            input_path: str,
            output_path: str
    ) -> None:
        npy_path = f"{input_path}/{sample.audio_file_name}.npy"
        csv_path = f"{output_path}/{sample.audio_file_name}.csv"

        make_directory(directory_name(csv_path), exist_ok=True)

        with open(npy_path, "rb") as infile:
            spectrogram = ndarray_load_from_disk(infile)

            cmap = ConstellationMap.from_spectrogram(
                spectrogram=spectrogram,
                threshold=threshold
            )

            fingerprints = cmap.fingerprints(
                label=sample.label,
                region_size=region_size,
                hash_func=Fingerprint.HASH_FUNCTION
            )

            with open(csv_path, "wt") as outfile:
                writer = CSVDictWriter(outfile, ["hash", "offset"])
                writer.writeheader()

                for fingerprint in fingerprints:
                    writer.writerow({
                        "hash": fingerprint.hash_value.hexdigest(),
                        "offset": fingerprint.offset
                    })

        print(f"DONE {csv_path}")

    def run(
            self,
            num_workers: int,
            region_size: int,
            threshold: float,
            dataset: Dataset
    ) -> None:
        with WorkerPool(num_workers) as pool:
            pool.map(
                partial(
                    FingerprintPreprocessor.do_work,
                    threshold=threshold,
                    region_size=region_size,
                    input_path=self._input_path,
                    output_path=self._output_path
                ),
                dataset
            )
