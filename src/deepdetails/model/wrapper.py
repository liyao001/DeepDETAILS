from typing import Optional, TypeAlias, Union

import pytorch_lightning as pl
import torch
from einops import rearrange

from deepdetails.helper.inspection import (
    bulk_visual_inspection,
    per_cluster_visual_inspection,
)
from deepdetails.helper.utils import (
    calc_counts_per_locus,
    transform_counts,
)
from deepdetails.model.deconvolution import Regressor, SeqOnlyRegressor
from deepdetails.model.loss import RMSLELoss, corrcoef_stable, mean_sq_offdiag_corr
from deepdetails.par_description import PARAM_DESC

# DeepDETAILSBatch: ((seq, acc), counts, profiles, per-cluster refs, loads, misc).
# per-cluster refs is [] when ground truth is unavailable.
# misc is (chroms, starts, ends, region_types, prior).
DeepDETAILSBatch: TypeAlias = tuple[
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


class DeepDETAILS(pl.LightningModule):
    def __init__(
        self,
        expected_clusters: int,
        profile_shrinkage: int = 1,
        filters: int = 512,
        n_non_dil_layers: int = 0,
        non_dil_kernel_size: int = 3,
        n_dil_layers: int = 8,
        dil_kernel_size: int = 3,
        conv1_kernel_size: int = 21,
        gru_layers: int = 1,
        gru_dropout: float = 0.1,
        profile_kernel_size: int = 75,
        head_mlp_layers: int = 3,
        num_tasks: int = 2,
        first_pass: Optional[bool] = None,
        redundancy_loss_coef: float = 0.01,
        prior_loss_coef: float = 1.0,
        learning_rate: float = 1e-3,
        version: str = "",
        lr_step_size: int = 1,
        lr_gamma: float = 0.1,
        scale_function_placement: str = "late-ch",
        t_x: int = 4096,
        test_screenshot_ratio: float = 0.002,
        gamma: float = 1e-8,
        rescaling_mode: int = 0,
        n_times_more_embeddings: int = 2,
        betas: tuple[float, float] = (0.9, 0.999),
        seq_only: Optional[bool] = False,
    ) -> None:
        """

        Parameters
        ----------
        expected_clusters : int
            {expected_clusters}
        profile_shrinkage : int
            {profile_shrinkage}
        filters : int
            {filters}
        n_non_dil_layers : int
            {n_non_dil_layers}
        non_dil_kernel_size : int
            {non_dil_kernel_size}
        n_dil_layers : int
            {n_dilated_layers}
        dil_kernel_size : int
            {dil_kernel_size}
        conv1_kernel_size : int
            {conv1_kernel_size}
        gru_layers : int
            {gru_layers}
        gru_dropout : float
            {gru_dropout}
        profile_kernel_size : int
            {profile_kernel_size}
        head_mlp_layers : int
            {head_mlp_layers}
        num_tasks : int
            {num_tasks}
        first_pass : Optional[bool]
            {first_pass}
        redundancy_loss_coef : float
            {redundancy_loss_coef}
        prior_loss_coef : float
            {prior_loss_coef}
        learning_rate : float
            {learning_rate}
        lr_step_size : int
            {lr_step_size}
        lr_gamma : float
            {lr_gamma}
        version : str
            {wandb_version}
        scale_function_placement : str
            {scale_function_placement}
        t_x : int
            {t_x}
        test_screenshot_ratio : float
            {test_screenshot_ratio}
        gamma : float
            {gamma}
        rescaling_mode : int
            {rescaling_mode}
        n_times_more_embeddings : int
            {n_times_more_embeddings}
        betas : tuple[float, float]
            {betas}
        seq_only : Optional[bool]
            {seq_only}
        """.format(**PARAM_DESC)
        super().__init__()
        self.save_hyperparameters()
        self.expected_clusters = expected_clusters
        self.example_input_array = (
            (
                torch.swapaxes(
                    torch.nn.functional.one_hot(torch.randint(0, 4, size=(8, t_x))),
                    1,
                    2,
                ).float(),
                torch.rand(8, expected_clusters, t_x),
            ),
            torch.nn.functional.softmax(torch.rand(8, expected_clusters), dim=0),
        )
        self.first_pass = first_pass
        self.model: Union[SeqOnlyRegressor, Regressor]

        if seq_only:
            self.model = SeqOnlyRegressor(
                expected_clusters=expected_clusters,
                filters=filters,
                n_non_dil_layers=n_non_dil_layers,
                non_dil_kernel_size=non_dil_kernel_size,
                n_dil_layers=n_dil_layers,
                dil_kernel_size=dil_kernel_size,
                conv1_kernel_size=conv1_kernel_size,
                profile_kernel_size=profile_kernel_size,
                counts_head_mlp_layers=head_mlp_layers,
                num_tasks=num_tasks,
                scale_function_placement=scale_function_placement,
            )
        else:
            self.model = Regressor(
                expected_clusters=expected_clusters,
                filters=filters,
                n_non_dil_layers=n_non_dil_layers,
                non_dil_kernel_size=non_dil_kernel_size,
                n_dil_layers=n_dil_layers,
                dil_kernel_size=dil_kernel_size,
                profile_shrinkage=profile_shrinkage,
                conv1_kernel_size=conv1_kernel_size,
                profile_kernel_size=profile_kernel_size,
                gru_layers=gru_layers,
                gru_dropout=gru_dropout,
                n_times_more_embeddings=n_times_more_embeddings,
                counts_head_mlp_layers=head_mlp_layers,
                num_tasks=num_tasks,
                scale_function_placement=scale_function_placement,
            )

        self.profile_loss_func = RMSLELoss()

        self.redundancy_loss_coef = redundancy_loss_coef
        self.prior_loss_coef = prior_loss_coef
        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.betas = betas
        self.version = version
        self.test_screenshot_ratio = test_screenshot_ratio
        self.gamma = gamma

        self.mod_rescaling = rescaling_mode
        self.self_qc_values = []
        self.sum_qc_metrics = torch.zeros(expected_clusters * num_tasks)
        self.enable_sum_qc_metrics = False

        # init lazy layers
        with torch.no_grad():
            example_input = self.example_input_array
            # pyrefly: ignore[not-iterable]
            example_x, example_loads = example_input
            self.forward(example_x, example_loads)

    def forward(self, x, loads):
        return self.model(x, loads)

    @staticmethod
    def _batch_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # per-batch r so Lightning's on_epoch mean stays sensitive to bad batches.
        x = torch.stack(
            [transform_counts(pred.flatten()), transform_counts(target.flatten())]
        )
        return corrcoef_stable(x)[0, 1]

    def training_step(self, batch: DeepDETAILSBatch, batch_idx: int):
        x, expected_counts, expected_profiles, _, loads, misc = batch
        batch_size = loads.shape[0]

        pc_profiles, pc_counts, _, _ = self.model(x, loads)

        cs_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)
        preds = cs_preds.sum(dim=0)

        msle_loss = self.profile_loss_func(preds, expected_profiles)
        self.log(
            "train_msle_loss",
            msle_loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=True,
        )

        reshaped = rearrange(cs_preds, "c b s l -> c b (s l)")
        reshaped = (
            reshaped + torch.arange(reshaped.shape[-1], device=self.device) * self.gamma
        )

        if reshaped.shape[1] > 1:
            # When the coefficient is 0, keep branch_corrs out of the graph
            branch_corrs = torch.stack(
                [mean_sq_offdiag_corr(sample) for sample in reshaped]
            ).mean()
            if batch_idx % 50 == 0:
                self.self_qc_values.append(branch_corrs.item())
        else:
            branch_corrs = msle_loss.new_tensor(0.0)
        self.log("train_br_cor", branch_corrs, batch_size=batch_size, on_step=True)

        if misc[-1].dim() == 2:
            prior = misc[-1][:, :]
            cluster_preds = rearrange(
                cs_preds
                + torch.arange(cs_preds.shape[-1], device=self.device) * self.gamma,
                "c b s l -> c (b s l)",
            )
            observed_corrs = corrcoef_stable(cluster_preds)
            prior_loss = (prior - observed_corrs).pow(2).mean()
            self.log(
                "train_prior_loss", prior_loss, batch_size=batch_size, on_step=True
            )
            loss = (
                msle_loss
                + branch_corrs * self.redundancy_loss_coef
                + prior_loss * self.prior_loss_coef
            )
        else:
            loss = msle_loss + branch_corrs * self.redundancy_loss_coef

        cor = self._batch_pearson(preds, expected_profiles)

        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log("train_corr", cor, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: DeepDETAILSBatch, batch_idx: int):
        x, expected_counts, expected_profiles, _, loads, misc = batch
        batch_size = loads.shape[0]

        pc_profiles, pc_counts, _, _ = self.model(x, loads)
        preds = calc_counts_per_locus(pc_profiles, pc_counts, False)

        msle_loss = self.profile_loss_func(preds, expected_profiles)
        self.log("val_msle_loss", msle_loss, batch_size=batch_size, prog_bar=True)
        loss = msle_loss

        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

        val_cor = self._batch_pearson(preds, expected_profiles)

        if torch.rand(1)[0] < 0.1:
            bulk_visual_inspection(
                preds,
                expected_profiles,
                calc_counts_per_locus(pc_profiles, pc_counts, True),
                f"e{self.current_epoch}.b{x[0].sum().item():.4f}.s",
                logger=self.logger,
            )

        self.log("val_corr", val_cor, batch_size=batch_size, prog_bar=True)

        return loss

    def _groundtruth_based_eval(
        self, per_cluster_y: torch.Tensor, per_cluster_y_hat: torch.Tensor
    ):
        """

        Parameters
        ----------
        per_cluster_y : torch.Tensor
            shape: (# cluster, batch, strand, seq_len)
        per_cluster_y_hat : torch.Tensor
            shape: (# cluster, batch, strand, seq_len)

        Returns
        -------

        """
        batch_size = per_cluster_y_hat.shape[1]

        y_hats_list = []
        if per_cluster_y.shape[0] == per_cluster_y_hat.shape[0]:
            for i, real_profiles in enumerate(per_cluster_y):
                y_hats = per_cluster_y_hat[i]
                y_hats_list.append(y_hats)
                test_cor = self._batch_pearson(y_hats, real_profiles)
                self.log(
                    f"test_corr_{i}", test_cor, batch_size=batch_size, on_epoch=True
                )

    def test_step(
        self, batch: DeepDETAILSBatch, batch_idx: int, dataloader_idx: int = 0
    ):
        (
            x,
            expected_counts,
            expected_profiles,
            expected_per_cluster_profiles,
            loads,
            _,
        ) = batch
        batch_size = loads.shape[0]

        pc_profiles, pc_counts, pc_weights, _ = self.model(x, loads)

        preds = calc_counts_per_locus(pc_profiles, pc_counts, False)
        cluster_preds = calc_counts_per_locus(
            pc_profiles, pc_counts, True
        )  # cluster, batch, strand, seq_len

        # routine evaluation
        msle_loss = self.profile_loss_func(preds, expected_profiles)

        test_cor = self._batch_pearson(preds, expected_profiles)

        # groundtruth-based evaluation
        if len(expected_per_cluster_profiles) > 0:
            stacked_refs = torch.stack(expected_per_cluster_profiles)
            self._groundtruth_based_eval(stacked_refs, cluster_preds)

            if torch.rand(1)[0] < self.test_screenshot_ratio:
                if isinstance(pc_weights, tuple):
                    pc_weights = torch.zeros(loads.shape[0], loads.shape[1])
                per_cluster_visual_inspection(
                    preds,
                    cluster_preds,
                    expected_profiles,
                    stacked_refs,
                    loads,
                    pc_weights,
                    f"preview{batch_idx}.{dataloader_idx}.{x[0].sum().item():.4f}.s",
                    logger=self.logger,
                )
        else:  # no groundtruth, only plot preds
            if torch.rand(1)[0] < self.test_screenshot_ratio:
                bulk_visual_inspection(
                    preds,
                    expected_profiles,
                    cluster_preds,
                    f"preview{batch_idx}.{dataloader_idx}.{x[0].sum().item():.4f}.s",
                    logger=self.logger,
                )

        self.log("test_loss", msle_loss, batch_size=batch_size, on_epoch=True)
        self.log(
            "test_corr", test_cor, batch_size=batch_size, prog_bar=True, on_epoch=True
        )

        per_cluster_per_strand_total = (
            torch.stack(pc_counts)
            .clone()
            .detach()
            .sum(dim=1)
            .flatten()
            .to(self.sum_qc_metrics.device)
        )
        self.sum_qc_metrics = self.sum_qc_metrics + per_cluster_per_strand_total
        self.enable_sum_qc_metrics = True

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=self.betas
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [lr_scheduler]
