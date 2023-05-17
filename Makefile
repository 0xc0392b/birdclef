install:
	pip install .

jupyter: install
	jupyter notebook ./notebooks

test-dataset: install
	python -m unittest tests.test_dataset

test-all: test-dataset
