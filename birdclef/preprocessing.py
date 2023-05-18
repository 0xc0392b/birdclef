from os import makedirs as make_directory
from os.path import dirname as directory_name
from multiprocessing import Pool as WorkerPool
from functools import partial

from numpy import save as save_ndarray

from .dataset import Dataset, Sample
from .feature_engineering import FeaturePipeline


class Preprocessor:
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
            save_ndarray(outfile, features)
            print(f"DONE {path}")

    def run(self, num_workers: int, dataset: Dataset) -> None:
        with WorkerPool(num_workers) as pool:
            pool.map(
                partial(
                    Preprocessor.do_work,
                    feature_pipeline=self._feature_pipeline,
                    sample_rate=self._sample_rate,
                    output_path=self._output_path
                ),
                dataset
            )
