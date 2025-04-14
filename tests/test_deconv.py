import unittest
import os
from deepdetails.protocols import deconv


class DeconvolutionTestCase(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.join(current_dir, "data")

    def test_deconv(self):
        """

        """
        try:
            for seq_only in [True, False]:
                deconv(
                    dataset=self.root, save_to=".", study_name="test", batch_size=8, num_workers=1, min_delta=1,
                    save_preds=True, chrom_cv=False, y_length=1000, earlystop_patience=1,
                    max_epochs=1, save_top_k_model=1, model_summary_depth=0, hide_progress_bar=True,
                    accelerator="cpu", devices="auto", version="", wandb_project=None, wandb_entity=None, gamma=0.0001,
                    wandb_upload_model=False, profile_shrinkage=8, filters=128, n_non_dil_layers=1,
                    non_dil_kernel_size=3, n_dilated_layers=9, dil_kernel_size=4, head_layers=1,
                    conv1_kernel_size=21, gru_layers=1, gru_dropout=0.1, profile_kernel_size=9,
                    redundancy_loss_coef=1., prior_loss_coef=1., rescaling_mode=1,
                    scale_function_placement="late-ch", learning_rate=0.001, betas=(0.9, 0.999),
                    all_regions=False, test_pos_only=True, cv=None, ct=None, max_retry=1,
                    seq_only=seq_only
                )
        except Exception as e:
            self.fail(f"Exception raised: {e}")


if __name__ == "__main__":
    unittest.main()
