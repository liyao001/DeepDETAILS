import glob
import os
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pytest

from deepdetails.cli import deepdetails
from deepdetails.helper.export import STRAND_LABELS
from deepdetails.helper.utils import slugify


@pytest.mark.train
class TrainPipelineTestCase(unittest.TestCase):
    """Train once, then cover deconv / export-pred / attr against the same checkpoint."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cls.root = os.path.join(current_dir, "data")
        cls.chrom_size = os.path.join(cls.root, "hg38.chrom.sizes")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.workdir = cls._tmpdir.name
        cls.ckpt = cls._train(
            cls.workdir,
            study_name="test",
            seq_only=False,
            no_preds=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        super().tearDownClass()

    @staticmethod
    def _run_cli(args):
        with mock.patch.object(sys, "argv", ["deepdetails", *map(str, args)]):
            deepdetails()

    @classmethod
    def _train(
        cls,
        save_to: str,
        *,
        study_name: str,
        seq_only: bool = False,
        no_preds: bool = True,
        require_ckpt: bool = True,
    ) -> str | None:
        # Hold out chroms absent from the fixture so all chr22 regions train.
        args = [
            "deconv",
            "--dataset",
            cls.root,
            "--save-to",
            save_to,
            "--study-name",
            study_name,
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--min-delta",
            "1",
            "--earlystop-patience",
            "1",
            "--max-epochs",
            "1",
            "--save-top-k-model",
            "1",
            "--model-summary-depth",
            "0",
            "--hide-progress-bar",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--chromosomal-validation",
            "chr1",
            "--chromosomal-testing",
            "chr2",
            "--gamma",
            "0.0001",
            "--profile-shrinkage",
            "8",
            "--filters",
            "128",
            "--head-mlp-layers",
            "1",
            "--gru-layers",
            "1",
            "--gru-dropout",
            "0.1",
            "--lr-step-size",
            "1",
            "--lr-gamma",
            "0.1",
            "--scale-function-placement",
            "late-ch",
            "--learning-rate",
            "0.001",
            "--betas",
            "0.9",
            "0.999",
            "--peak-only",
            "--max-retry",
            "1",
        ]
        if seq_only:
            args.append("--seq")
        if no_preds:
            args.append("--no-preds")
        cls._run_cli(args)
        ckpts = sorted(
            glob.glob(os.path.join(save_to, "**", "*.ckpt"), recursive=True)
        )
        if not ckpts:
            if require_ckpt:
                raise AssertionError(f"No checkpoint written under {save_to}")
            return None
        return ckpts[0]

    def test_deconv(self):
        self.assertTrue(os.path.exists(self.ckpt), self.ckpt)

    def test_deconv_seq_only(self):
        with tempfile.TemporaryDirectory() as save_to:
            self._train(
                save_to,
                study_name="test-seq",
                seq_only=True,
                require_ckpt=False,
            )

    def test_export_pred(self):
        export_dir = os.path.join(self.workdir, "export-pred")
        os.makedirs(export_dir, exist_ok=True)
        self._run_cli(
            [
                "export-pred",
                "--dataset",
                self.root,
                "--checkpoint",
                self.ckpt,
                "--save-to",
                export_dir,
                "--study-name",
                "test",
                "--batch-size",
                "8",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--hide-progress-bar",
                "--all-regions",
            ]
        )

        pred_file = os.path.join(export_dir, "test.predictions.h5")
        counts_file = os.path.join(export_dir, "test.counts.csv.gz")
        self.assertTrue(os.path.exists(pred_file), pred_file)
        self.assertTrue(os.path.exists(counts_file), counts_file)

        with h5py.File(pred_file, "r") as fh:
            self.assertIn("preds", fh)
            self.assertIn("regions", fh)
            preds = fh["preds"]
            self.assertEqual(preds.ndim, 4)
            # (n_clusters, n_regions, n_targets, y_length)
            self.assertEqual(preds.shape[0], 3)
            self.assertGreater(preds.shape[1], 0)
            self.assertEqual(preds.shape[2], 2)
            self.assertEqual(preds.shape[3], 1000)
            self.assertFalse(_all_nan_or_zero(preds[:]))
            n_clusters = int(preds.attrs["n_clusters"])
            n_strands = int(preds.attrs["n_targets"])
            cluster_names = list(preds.attrs["cluster_names"])

        bw_dir = os.path.join(export_dir, "bw")
        os.makedirs(bw_dir, exist_ok=True)
        self._run_cli(
            [
                "build-bw",
                "--pred-file",
                pred_file,
                "--save-to",
                bw_dir,
                "--chrom-size",
                self.chrom_size,
                "--num-workers",
                "1",
                # Keep nearly all predicted values so bedGraphs are non-empty.
                "--min-abs-val",
                "0",
            ]
        )

        self.assertEqual(n_strands, len(STRAND_LABELS))
        self.assertEqual(len(cluster_names), n_clusters)
        for cluster_name in cluster_names:
            safe_name = slugify(str(cluster_name))
            for strand in STRAND_LABELS:
                bw_path = os.path.join(bw_dir, f"{safe_name}.{strand}.bw")
                self.assertTrue(os.path.exists(bw_path), bw_path)
                self.assertGreater(os.path.getsize(bw_path), 0)

    def test_attr(self):
        attr_dir = os.path.join(self.workdir, "attr")
        os.makedirs(attr_dir, exist_ok=True)
        # ReducedDataset writes region_mapping.bed into the process CWD.
        prev_cwd = os.getcwd()
        try:
            os.chdir(attr_dir)
            self._run_cli(
                [
                    "attr",
                    "--dataset",
                    self.root,
                    "--checkpoint",
                    self.ckpt,
                    "--save-to",
                    attr_dir,
                    "--batch-size",
                    "4",
                    "--device",
                    "cpu",
                    "--chrom-size",
                    self.chrom_size,
                    "--hide-progress-bar",
                ]
            )
        finally:
            os.chdir(prev_cwd)

        attr_file = os.path.join(attr_dir, "attr.h5")
        self.assertTrue(os.path.exists(attr_file), attr_file)
        self.assertTrue(os.path.exists(os.path.join(attr_dir, "region_mapping.bed")))

        with h5py.File(attr_file, "r") as fh:
            self.assertIn("ohe", fh)
            self.assertIn("contrib", fh)
            ohe = fh["ohe"]
            contrib = fh["contrib"]
            self.assertEqual(ohe.ndim, 3)  # (n_samples, 4, seq_len)
            self.assertEqual(contrib.ndim, 4)  # (n_samples, n_clusters, 4, seq_len)
            self.assertEqual(ohe.shape[1], 4)
            self.assertEqual(contrib.shape[1], 3)
            self.assertEqual(contrib.shape[2], 4)
            self.assertEqual(ohe.shape[0], contrib.shape[0])
            self.assertGreater(ohe.shape[0], 0)
            self.assertFalse(_all_nan_or_zero(contrib[:]))

        for cluster_idx in range(3):
            bw_path = os.path.join(attr_dir, f"C{cluster_idx}.bw")
            self.assertTrue(os.path.exists(bw_path), bw_path)
            self.assertGreater(os.path.getsize(bw_path), 0)


def _all_nan_or_zero(arr) -> bool:
    a = np.asarray(arr)
    return bool(np.isnan(a).all() or np.allclose(a, 0.0))


if __name__ == "__main__":
    unittest.main()
