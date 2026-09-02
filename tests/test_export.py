"""Unit tests for bedGraph → bigWig export (native sort | merge | awk pipeline)."""

import os
import shutil
import tempfile
import unittest

import pytest
import pyBigWig

from deepdetails.helper.export import bg_to_bw_core
from deepdetails.helper.utils import run_pipeline

_EXPORT_TOOLS = ("awk", "sort", "bedtools", "bedGraphToBigWig")
_MISSING_TOOLS = [tool for tool in _EXPORT_TOOLS if shutil.which(tool) is None]


def _write_lines(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.mark.unit
@unittest.skipUnless(not _MISSING_TOOLS, f"missing tools: {', '.join(_MISSING_TOOLS)}")
class BgToBwCoreTestCase(unittest.TestCase):
    def test_sorts_merges_overlaps_and_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "out dir")
            os.makedirs(work)
            bg = os.path.join(work, "in file.bg")
            chrom_sizes = os.path.join(work, "chrom.sizes")
            _write_lines(
                bg,
                "chr1\t15\t25\t2.5\nchr2\t0\t10\t4.0\nchr1\t10\t20\t1.5\n",
            )
            _write_lines(chrom_sizes, "chr1\t1000\nchr2\t1000\n")

            dest = bg_to_bw_core(bg, os.path.join(work, "signal"), chrom_sizes, coef=-1)

            self.assertTrue(os.path.exists(dest))
            self.assertFalse(os.path.exists(bg))
            with pyBigWig.open(dest) as bw:
                self.assertAlmostEqual(bw.stats("chr1", 10, 25)[0], -2.0)
                self.assertAlmostEqual(bw.stats("chr2", 0, 10)[0], -4.0)

    def test_skip_sort_merge_scales_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            bg = os.path.join(tmp, "sorted.bg")
            chrom_sizes = os.path.join(tmp, "chrom.sizes")
            _write_lines(bg, "chr1\t10\t20\t1.5\nchr1\t30\t40\t3.0\n")
            _write_lines(chrom_sizes, "chr1\t1000\n")

            dest = bg_to_bw_core(
                bg,
                os.path.join(tmp, "scaled"),
                chrom_sizes,
                coef=-1,
                skip_sort_merge=True,
            )

            with pyBigWig.open(dest) as bw:
                self.assertAlmostEqual(bw.stats("chr1", 10, 20)[0], -1.5)
                self.assertAlmostEqual(bw.stats("chr1", 30, 40)[0], -3.0)


@pytest.mark.unit
class RunPipelineTestCase(unittest.TestCase):
    def test_raises_on_nonzero_exit(self):
        with tempfile.TemporaryFile() as out:
            with self.assertRaises(RuntimeError) as caught:
                run_pipeline([["false"]], out)
        self.assertIn("false failed", str(caught.exception))
