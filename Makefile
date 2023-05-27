install:
	pip install .

jupyter: install
	jupyter notebook ./notebooks

test-dataset: install
	python -m unittest tests.test_dataset

test-feature-engineering: install
	python -m unittest tests.test_feature_engineering

test-classifier: install
	python -m unittest tests.test_classifier

test-acoustic-fingerprint: install
	python -m unittest tests.test_acoustic_fingerprint

test-soundscape: install
	python -m unittest tests.test_soundscape

test-spectrogram-preprocessing: install
	python -m unittest tests.test_preprocessing.TestSpectrogramPreprocessor

test-peak-preprocessing: install
	python -m unittest tests.test_preprocessing.TestPeakPreprocessor

test-fingerprint-preprocessing: install
	python -m unittest tests.test_preprocessing.TestFingerprintPreprocessor
