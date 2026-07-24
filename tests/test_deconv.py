import unittest
import os
import sys
import tempfile
from unittest import mock

import pytest

from deepdetails.cli import deepdetails


@pytest.mark.train
class DeconvolutionTestCase(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(current_dir, "data")

    def test_deconv(self):
        for seq_only in [True, False]:
            with tempfile.TemporaryDirectory() as save_to:
                args = [
                    "deepdetails", "deconv",
                    "--dataset", self.root,
                    "--save-to", save_to,
                    "--study-name", "test",
                    "--batch-size", "8",
                    "--num-workers", "1",
                    "--min-delta", "1",
                    "--earlystop-patience", "1",
                    "--max-epochs", "1",
                    "--save-top-k-model", "1",
                    "--model-summary-depth", "0",
                    "--hide-progress-bar",
                    "--accelerator", "cpu",
                    "--devices", "1",
                    "--gamma", "0.0001",
                    "--profile-shrinkage", "8",
                    "--filters", "128",
                    "--head-mlp-layers", "1",
                    "--gru-layers", "1",
                    "--gru-dropout", "0.1",
                    "--lr-step-size", "1",
                    "--lr-gamma", "0.1",
                    "--scale-function-placement", "late-ch",
                    "--learning-rate", "0.001",
                    "--betas", "0.9", "0.999",
                    "--peak-only",
                    "--max-retry", "1",
                ]
                if seq_only:
                    args.append("--seq")

                with mock.patch.object(sys, "argv", args):
                    deepdetails()


if __name__ == "__main__":
    unittest.main()
