"""Regression tests for attribution bigWig export ordering."""

import os
import tempfile
import unittest

import h5py
import numpy as np
import pyBigWig

from deepdetails.helper.attr import write_scores_to_bigwigs


class WriteScoresToBigwigsTestCase(unittest.TestCase):
    def test_writes_chromosomes_in_chrom_sizes_order(self):
        # peaks are lex-ordered (chr1, chr10, chr2); header is chr1, chr2, chr10
        with tempfile.TemporaryDirectory() as tmp:
            chrom_sizes = os.path.join(tmp, "chrom.sizes")
            with open(chrom_sizes, "w", encoding="utf-8") as fh:
                fh.write("chr1\t1000\nchr2\t1000\nchr10\t1000\n")

            peaks = os.path.join(tmp, "peaks.bed")
            with open(peaks, "w", encoding="utf-8") as fh:
                fh.write("chr1\t10\t20\n")
                fh.write("chr10\t10\t20\n")
                fh.write("chr2\t10\t20\n")

            score_file = os.path.join(tmp, "attr.h5")
            # contrib shape: (n_regions, n_clusters, 4, seq_len)
            contrib = np.ones((3, 1, 4, 10), dtype=np.float32)
            with h5py.File(score_file, "w") as fh:
                fh.create_dataset("contrib", data=contrib)

            out_bw = os.path.join(tmp, "C0.bw")
            write_scores_to_bigwigs(score_file, peaks, 0, out_bw, chrom_sizes)

            self.assertTrue(os.path.exists(out_bw))
            self.assertGreater(os.path.getsize(out_bw), 0)
            with pyBigWig.open(out_bw) as bw:
                self.assertAlmostEqual(bw.stats("chr1", 10, 20)[0], 4.0)
                self.assertAlmostEqual(bw.stats("chr2", 10, 20)[0], 4.0)
                self.assertAlmostEqual(bw.stats("chr10", 10, 20)[0], 4.0)


if __name__ == "__main__":
    unittest.main()
