from unittest import TestCase

from birdclef import Sample, Dataset


class TestSample(TestCase):
    def setUp(self):
        self._sample = Sample(
            dataset_path=Dataset.PATH,
            audio_directory=Dataset.AUDIO_DIRECTORY,
            audio_file_path="abethr1/XC128013.ogg",
            label="abethr1",
            author="Rolf A. de By",
            longitude=38.2788,
            latitude=4.3906
        )

    def test_label(self):
        self.assertEqual(self._sample.label, "abethr1")

    def test_author(self):
        self.assertEqual(self._sample.author, "Rolf A. de By")

    def test_coordinates(self):
        self.assertEqual(self._sample.coordinates, (38.2788, 4.3906))

    def test_audio_file_path(self):
        path = f"{Dataset.PATH}/{Dataset.AUDIO_DIRECTORY}/abethr1/XC128013.ogg"
        self.assertEqual(self._sample.audio_file_path, path)

    def test_audio_samples(self):
        samples = self._sample.audio_samples(Dataset.SAMPLE_RATE)
        self.assertEqual(len(samples), 1459513)


class TestDataset(TestCase):
    def setUp(self):
        self._dataset = Dataset.load(Dataset.PATH)

    def test_load(self):
        self.assertEqual(len(self._dataset), 16941)

    def test_labels(self):
        self.assertEqual(len(self._dataset.labels()), 264)

    def test_pick_random(self):
        self.assertTrue(self._dataset.pick_random() != None)

    def test_with_label(self):
        equal = map(lambda x: x.label == "labnel", self._dataset.with_label("labnel"))
        self.assertTrue(all(equal))
