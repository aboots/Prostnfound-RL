import argparse
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange


def simple_mil_loss(
    patch_logits,
    patch_labels,
    core_indices,
    top_percentile=0.2,
    pos_weight=torch.tensor(1.0),
):
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none"
    )

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)

    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_patches_to_keep = int(n_patches * top_percentile)
        patch_losses_for_core_sorted = torch.sort(patch_losses_for_core)[0]
        patch_losses_for_core_to_keep = patch_losses_for_core_sorted[:n_patches_to_keep]
        loss += patch_losses_for_core_to_keep.mean()

    return loss


class CancerDetectionLossBase(nn.Module):
    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        raise NotImplementedError


class CancerDetectionValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        base_loss: str = "ce",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        loss = torch.tensor(0, dtype=torch.float32, device=predictions.device)
        if self.base_loss == "ce":
            loss += nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
        elif self.base_loss == "gce":
            # we should convert to "two class" classification problem
            loss_fn = BinaryGeneralizedCrossEntropy()
            loss += loss_fn(predictions, labels)
        elif self.base_loss == "mae":
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels, reduction="none"
            )
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        return loss


class CancerDetectionSoftValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        loss_pos_weight: float = 1,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        sigma: float = 15,
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.sigma = sigma

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement, **kwargs):
        masks = []
        for i in range(len(cancer_logits)):
            mask = prostate_mask[i] > 0.5
            mask = mask & (needle_mask[i] > 0.5)
            mask = mask.float().cpu().numpy()[0]

            # resize and blur mask
            from skimage.transform import resize

            mask = resize(mask, (256, 256), order=0)
            from skimage.filters import gaussian

            mask = gaussian(mask, self.sigma, mode="constant", cval=0)
            mask = mask - mask.min()
            mask = mask / mask.max()
            mask = torch.tensor(mask, device=cancer_logits.device)[None, ...]

            masks.append(mask)
        masks = torch.stack(masks)

        B = label.shape[0]
        label = label.repeat(B, 1, 256, 256).float()
        loss_by_pixel = nn.functional.binary_cross_entropy_with_logits(
            cancer_logits,
            label,
            pos_weight=torch.tensor(self.loss_pos_weight, device=cancer_logits.device),
            reduction="none",
        )
        loss = (loss_by_pixel * masks).mean()
        return loss


class MultiTermCanDetLoss(CancerDetectionLossBase):
    def __init__(self, loss_terms: list[CancerDetectionLossBase], weights: list[float]):
        super().__init__()
        self.loss_terms = loss_terms
        self.weights = weights

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement, **kwargs):
        loss = torch.tensor(0, dtype=torch.float32, device=cancer_logits.device)
        for term, weight in zip(self.loss_terms, self.weights):
            loss += weight * term(
                cancer_logits, prostate_mask, needle_mask, label, involvement
            )
        return loss


class BinaryGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = pred.sigmoid()[..., 0]
        labels = labels[..., 0].long()
        pred = torch.stack([1 - pred, pred], dim=-1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, 2).float().to(pred.device)
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            # core is benign, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        elif involvement_for_core.item() > 0.65:
            # core is high involvement, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        else:
            # core is of intermediate involvement, so label noise is assumed to be high.
            # we should be tolerant of the model's "false positives" in this case.
            pred_index_sorted_by_cancer_score = torch.argsort(
                patch_logits_for_core[:, 0], descending=True
            )
            patch_logits_for_core = patch_logits_for_core[
                pred_index_sorted_by_cancer_score
            ]
            patch_labels_for_core = patch_labels_for_core[
                pred_index_sorted_by_cancer_score
            ]
            n_predictions = patch_logits_for_core.shape[0]
            patch_predictions_for_core_for_loss = patch_logits_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            patch_labels_for_core_for_loss = patch_labels_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_predictions_for_core_for_loss,
                patch_labels_for_core_for_loss,
            )


def build_loss(args):
    # setup criterion
    loss_terms = []
    loss_weights = []

    for i in range(len(args.loss_name)):
        loss_name = args.loss_name[i]
        base_loss_name = args.base_loss_name[i]
        loss_pos_weight = args.loss_pos_weight[i]
        loss_prostate_mask = args.loss_prostate_mask[i]
        loss_needle_mask = args.loss_needle_mask[i]
        loss_weight = args.loss_weight[i]

        if loss_name == "valid_region":
            loss_terms.append(
                CancerDetectionValidRegionLoss(
                    base_loss=base_loss_name,
                    loss_pos_weight=loss_pos_weight,
                    prostate_mask=loss_prostate_mask,
                    needle_mask=loss_needle_mask,
                )
            )
            loss_weights.append(loss_weight)
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

    return MultiTermCanDetLoss(loss_terms, loss_weights)


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    group = parser.add_argument_group("Loss")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    group.add_argument(
        "--loss_name", nargs="+", choices=("valid_region",), default=("valid_region",)
    )
    group.add_argument(
        f"--base_loss_name",
        type=str,
        default=["ce"],
        choices=("ce", "gce", "mae", "mil"),
        nargs="+",
        help="""The name of the lower-level loss function to use. Our experiments showed best performance with simple CE loss, but 
        due to weak supervision and label noise, we also experimented with GCE, MAE, and MIL. However, we found no benefit beyond using CE loss.""",
    )
    group.add_argument(
        f"--loss_pos_weight",
        type=float,
        default=[1.0],
        help="""The positive class weight for the loss function. If using a large ratio of benign to cancer cores in training, it is recommended to increase this value to 2 or 3 to account for the class imbalance.""",
        nargs="+",
    )
    group.add_argument(
        f"--loss_prostate_mask",
        type=str2bool,
        default=[True],
        help="If True, the loss will only be applied inside the prostate mask.",
        nargs="+",
    )
    group.add_argument(
        f"--loss_needle_mask",
        type=str2bool,
        default=[True],
        help="If True, the loss will only be applied inside the needle mask.",
        nargs="+",
    )
    group.add_argument(
        f"--loss_weight",
        type=float,
        default=[1.0],
        help="The weight to use for the loss function.",
        nargs="+",
    )

    return parser


