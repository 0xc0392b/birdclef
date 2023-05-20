from unittest import TestCase

from numpy import array
from numpy.random import rand as rand_array

from birdclef import ConstellationMap, CandidatePeak, Fingerprint


class TestFingerprint(TestCase):
    def setUp(self):
        self._fp1 = Fingerprint.from_pair(
            a=CandidatePeak(100, 1000000),
            b=CandidatePeak(200, 2000000),
            hash_func=Fingerprint.HASH_FUNCTION,
            offset=1,
            label="test"
        )

        self._fp2 = Fingerprint.from_pair(
            a=CandidatePeak(100, 3000000),
            b=CandidatePeak(200, 4000000),
            hash_func=Fingerprint.HASH_FUNCTION,
            offset=2,
            label="test"
        )

    def test_hash_value(self):
        h1 = self._fp1.hash_value
        h2 = self._fp2.hash_value
        self.assertNotEqual(h1, h2)

    def test_offset(self):
        self.assertEqual(self._fp1.offset, 1)
        self.assertEqual(self._fp2.offset, 2)

    def test_label(self):
        self.assertEqual(self._fp1.label, "test")
        self.assertEqual(self._fp2.label, "test")

    def test_from_pair(self):
        self.assertNotEqual(str(self._fp1), str(self._fp2))


class TestConstellationMap(TestCase):
    def test_to_vectors(self):
        cmap = ConstellationMap([
            CandidatePeak(1, 100),
            CandidatePeak(2, 200),
            CandidatePeak(3, 300)
        ])

        times, frequencies = cmap.to_vectors()

        self.assertTrue(all(times == array([1, 2, 3])))
        self.assertTrue(all(frequencies == array([100, 200, 300])))

    def test_fingerprints(self):
        pass

    def test_from_spectrogram(self):
        cmap = ConstellationMap.from_spectrogram(rand_array(100, 1000))

        # ...
