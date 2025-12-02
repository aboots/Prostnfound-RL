from collections import defaultdict
import torch
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.utils.accumulators import DataFrameCollector
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2 as T
from matplotlib import pyplot as plt
from PIL import Image


def _auc_roc(predictions, labels):
    nanvalues = np.isnan(predictions)
    predictions = predictions[~nanvalues]
    labels = labels[~nanvalues]
    return roc_auc_score(labels, predictions)


@torch.no_grad()
def show_heatmap_prediction(data):

    plt.close("all")
    plt.figure()

    if "cancer_logits" in data:
        logits = data["cancer_logits"].cpu()
        pred = logits.sigmoid()
    elif "cancer_probs" in data:
        pred = data["cancer_probs"].cpu()
    else:
        raise ValueError()

    needle_mask = data["needle_mask"]
    prostate_mask = data["prostate_mask"]
    image = data["bmode"]
    label = data["label"]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    [ax.set_axis_off() for ax in ax.flatten()]
    kwargs = dict(vmin=0, vmax=1)

    image = T.Resize(
        (224, 224), interpolation=Image.Resampling.BICUBIC, antialias=True
    )(image)
    needle_mask = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(
        needle_mask
    )
    prostate_mask = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(
        prostate_mask
    )
    pred = T.Resize((224, 224), interpolation=Image.Resampling.NEAREST)(pred)

    # image and contours
    ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
    ax[0].contour(prostate_mask[0, 0], **kwargs)
    ax[0].contour(needle_mask[0, 0], **kwargs)

    # prediction
    ax[1].imshow(pred[0, 0], **kwargs)
    ax[1].contour(needle_mask[0, 0], **kwargs)
    ax[1].contour(prostate_mask[0, 0], **kwargs)

    # Build title with ground truth and model predictions
    gt_cancer = label[0].item() == 1
    involvement = data["involvement"][0].item()
    grade_group = data["grade_group"][0]

    # Heatmap-based prediction (needle average)
    pred_heatmap = None
    if "average_needle_heatmap_value" in data:
        pred_heatmap = float(data["average_needle_heatmap_value"][0].item())

    # Image-level classifier prediction (if available)
    cls_prob = None
    if "image_level_classification_outputs" in data:
        cls_outputs = data["image_level_classification_outputs"][0].detach().cpu()
        cls_probs = cls_outputs.softmax(-1)
        # Assume binary classification, cancer is class 1
        if cls_probs.ndim == 1:
            cls_prob = float(cls_probs[1].item())
        else:
            cls_prob = float(cls_probs[0, 1].item())

    title_parts = [
        f"GT: Cancer {gt_cancer}",
        f"Inv {involvement:.2f}",
        f"Grade group {grade_group}",
    ]
    if pred_heatmap is not None:
        title_parts.append(f"Heatmap p(cancer) {pred_heatmap:.2f}")
    if cls_prob is not None:
        title_parts.append(f"Image-level p(cancer) {cls_prob:.2f}")

    fig.suptitle("; ".join(title_parts))

    return fig


class CancerLogitsHeatmapsEvaluator:
    def __init__(
        self,
        log_images=False,
        log_images_every=10,
        include_patient_metrics=False,
        include_heatmap_cspca_metrics=True,
    ):
        self.iter = 0
        self.log_images = log_images
        self.log_images_every = log_images_every
        self.include_patient_metrics = include_patient_metrics
        self.accumulator = DataFrameCollector()
        self._heatmap_fig = None
        self.include_heatmap_cspca_metrics = include_heatmap_cspca_metrics
        self.results_table = None

    @torch.no_grad()
    def __call__(self, data):
        step_metrics = {}

        if "cancer_logits" in data:
            bags_of_logits = get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
            bags_of_probs = [bag.sigmoid() for bag in bags_of_logits]
        elif "cancer_probs" in data:
            bags_of_probs = get_bags_of_predictions(
                data["cancer_probs"], data["prostate_mask"], data["needle_mask"]
            )

        bag_level_info = defaultdict(list)

        for probs in bags_of_probs:
            # entropy
            normalized_probs = probs / probs.sum()
            entropy = -(normalized_probs * normalized_probs.log()).sum()
            bag_level_info["entropy"].append(entropy.item())

            # topk score
            N = len(probs)
            k = int(N * 0.5)
            topk_score = torch.sort(probs, descending=True).values[:k].mean()
            bag_level_info["topk_score"].append(topk_score.item())

        tracked_data = {}
        keys = [
            "center",
            "core_id",
            "patient_id",
            "loc",
            "grade",
            "age",
            "family_history",
            "psa",
            "pct_cancer",
            "grade_group",
            "average_needle_heatmap_value",
            "average_prostate_heatmap_value",
            "label",
            "involvement",
            "clinically_significant",
        ]
        for key in keys:
            tracked_data[key] = data[key]
        tracked_data.update(bag_level_info)

        if data.get("image_level_classification_outputs"):
            tracked_data["image_level_cancer_logits"] = (
                data["image_level_classification_outputs"][0]
                .detach()
                .cpu()
                .softmax(-1)[:, 1]
            )

        self.accumulator(tracked_data)

        if self.log_images and (self.iter % self.log_images_every == 0):
            figure = show_heatmap_prediction(data)
            step_metrics["heatmap_example"] = figure

        self.iter += 1
        return step_metrics

    def aggregate_metrics(self, results_table=None):
        from src.utils import calculate_metrics
        from sklearn.metrics import roc_auc_score

        results_table = results_table or self.accumulator.compute()
        self.results_table = results_table

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(predictions, labels, log_images=self.log_images)
        metrics.update(metrics_)

        metrics["topk_probs_auroc"] = _auc_roc(results_table.topk_score, labels)
        metrics["avg_bag_entropy"] = results_table["entropy"].mean()

        # prop pred err
        metrics["prop_pred_err"] = np.abs(
            results_table["average_needle_heatmap_value"].values
            - results_table["involvement"]
        ).mean()

        # balanced prop pred err
        results_table["prop_pred_err"] = (
            results_table["average_needle_heatmap_value"] - results_table["involvement"]
        ).abs()
        metrics["bal_prop_pred_err"] = (
            results_table.query("label == 0")["prop_pred_err"].mean()
            + results_table.query("label == 1")["prop_pred_err"].mean()
        ) / 2

        # balanced prob pred err using thresholded probabilities
        # prop_pred_err_t30 = (
        #     results_table.query("mean_binarized_act_t=30 > 0.5")["involvement"]
        #     - results_table.query("mean_binarized_act_t=30 > 0.5")[
        #         "average_needle_heatmap_value"
        #     ]
        # ).abs().mean()

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(
                core_probs, core_labels, log_images=self.log_images
            )
            metrics.update(
                {
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                }
            )
            metrics["topk_probs_auroc_high_inv"] = _auc_roc(
                results_table.topk_score.values[keep], core_labels
            )

        # patient predictions
        if self.include_patient_metrics:
            predictions = (
                results_table.groupby("patient_id")
                .average_prostate_heatmap_value.mean()
                .values
            )
            labels = (
                results_table.groupby("patient_id").clinically_significant.sum() > 0
            ).values
            metrics_ = calculate_metrics(
                predictions, labels, log_images=self.log_images
            )
            metrics.update(
                {f"{metric}_patient": value for metric, value in metrics_.items()}
            )

        if "image_level_cancer_logits" in results_table.columns:
            image_level_predictions = results_table.image_level_cancer_logits.values
            image_level_labels = results_table.label.values
            metrics_ = calculate_metrics(
                image_level_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {f"{metric}_image_level": value for metric, value in metrics_.items()}
            )

            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            metrics_low_vs_high = metrics_ = calculate_metrics(
                image_level_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {
                    f"{metric}_image_level_cspca": value
                    for metric, value in metrics_low_vs_high.items()
                }
            )

        if self.include_heatmap_cspca_metrics:
            heatmap_predictions = results_table["average_needle_heatmap_value"]
            image_level_labels = (results_table.grade_group.values > 2).astype(int)
            metrics_ = calculate_metrics(
                heatmap_predictions, image_level_labels, log_images=self.log_images
            )
            metrics.update(
                {
                    f"{metric}_heatmap_cspca": value
                    for metric, value in metrics_.items()
                }
            )

        return metrics

