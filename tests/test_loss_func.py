import unittest
import torch
import numpy as np
from scipy.special import softmax
from deepdetails.model.loss import RMSLELoss


class ProfileLossTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.target_shape = (3, 2, 100)  # batch, n_tasks, seq_length
        self.simulated_signals = torch.zeros(self.target_shape)

        # Simulate prediction targets
        # the first sample comes from two multinomial distributions
        rng = np.random.default_rng()
        self.simulated_signals[0, 0, :] = torch.from_numpy(
            rng.multinomial(200, softmax(np.random.randint(0, 50, 100)), size=1)
        )
        self.simulated_signals[0, 1, :] = torch.from_numpy(
            rng.multinomial(200, [1/100.]*self.target_shape[2], size=1)
        )
        
        # the second sample comes from |sin| or |cos| * 100, casted to the nearest integers
        self.simulated_signals[1, 0, :] = torch.from_numpy(
            np.abs((100 * np.sin(np.linspace(-np.pi, np.pi, 100))).astype(int))
        )
        self.simulated_signals[1, 1, :] = torch.from_numpy(
            np.abs((100 * np.cos(np.linspace(-np.pi, np.pi, 100))).astype(int))
        )

        # the third sample comes from two gaussian distributions
        self.simulated_signals[2, 0, :] = torch.from_numpy(
            np.histogram(np.random.default_rng().normal(50, 5, 1000), bins=100, range=(0, 100))[0]
        )
        self.simulated_signals[2, 1, :] = torch.from_numpy(
            np.histogram(np.random.default_rng().normal(30, 10, 1000), bins=100, range=(0, 100))[0]
        )
        
        # Simulate predictions
        # The first set of predictions captures the shape, but the values are scaled by a constant to the real vaules.
        # Expecting overall small loss, and loss for squash_05, stretch_2 should be smaller than that of squash_02 and stretch_3.
        self.preds_squash_02 = self.simulated_signals * 0.2
        self.preds_squash_05 = self.simulated_signals * 0.5
        self.preds_stretch_2 = self.simulated_signals * 2
        self.preds_stretch_3 = self.simulated_signals * 3

        # The second set of predictions just captured random noise, expecting the loss to be huge
        self.preds_rand = torch.rand(self.target_shape)

    def test_rmsle(self):
        loss_func = RMSLELoss()

        squash_02_loss = loss_func(self.preds_squash_02, self.simulated_signals)
        squash_05_loss = loss_func(self.preds_squash_05, self.simulated_signals)
        stretch_2_loss = loss_func(self.preds_stretch_2, self.simulated_signals)
        stretch_3_loss = loss_func(self.preds_stretch_3, self.simulated_signals)
        random_loss = loss_func(self.preds_rand, self.simulated_signals)

        self.assertTrue(squash_05_loss < squash_02_loss)
        self.assertTrue(stretch_2_loss < stretch_3_loss)
        self.assertTrue(squash_02_loss < random_loss)
        self.assertTrue(squash_05_loss < random_loss)
        self.assertTrue(stretch_2_loss < random_loss)
        self.assertTrue(stretch_3_loss < random_loss)
