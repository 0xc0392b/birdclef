install:
	pip install .

jupyter: install
	jupyter notebook ./notebooks

test-dataset: install
	python -m unittest tests.test_dataset

test-feature-engineering: install
	python -m unittest tests.test_feature_engineering

test-all: test-dataset test-feature-engineering
