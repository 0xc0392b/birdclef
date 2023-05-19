install:
	pip install .

jupyter: install
	jupyter notebook ./notebooks

test-dataset: install
	python -m unittest tests.test_dataset

test-feature-engineering: install
	python -m unittest tests.test_feature_engineering

test-preprocessing: install
	python -m unittest tests.test_preprocessing

test-classifier: install
	python -m unittest tests.test_classifier

test-all: test-dataset test-feature-engineering test-preprocessing test-classifier
