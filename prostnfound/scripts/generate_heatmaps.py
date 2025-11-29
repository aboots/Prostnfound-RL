import argparse
from functools import partial
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
from medAI.modeling import create_model
from argparse import Namespace
import numpy as np
import skimage
import sys
import os
from matplotlib import pyplot as plt


sys.path.append(os.getcwd())
from train import ProstNFoundMeta
from src.loaders import get_dataloaders


PROSTNFOUND_PLUS_FINAL_BINS = [
    0,
    0.017005179575169737,
    0.19308170455868087,
    0.2734902565179437,
    0.3382657863627546,
    0.5821527,
]


def main(
    checkpoint,
    output_dir,
    device="cpu",
    apply_prostate_mask=False,
    patient_ids=None,
    core_ids=None,
    mode=None,
    style=None,
    max_num=None,
    start_alpha=0,
    stop_alpha=1,
    alpha_cutoff=0.1,
    dataset_cfg_file=None,
    no_title=False,
    format='png', 
    dpi=300,
):

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output_dir = output_dir

    train_args = Namespace(**state["args"])
    model = ProstNFoundMeta(create_model(train_args.model, **train_args.model_kw))
    model.load_state_dict(state["model"])
    model.eval().to(device)

    train_args.num_workers = 0
    train_args.dataset = "optimum"

    if dataset_cfg_file is not None:
        data_cfg = OmegaConf.load(dataset_cfg_file)
    elif hasattr(train_args, "data"):
        data_cfg = train_args.data
    else:
        data_cfg = train_args
    loaders = get_dataloaders(data_cfg, mode="heatmap")

    from torch.utils.data import default_collate

    val_loader = loaders["val"]
    # batch = next(iter(val_loader))

    from copy import deepcopy

    val_dataset = val_loader.dataset
    val_dataset_raw = deepcopy(val_dataset)
    val_dataset_raw.transform = None

    def get_batch(i):
        item = val_dataset[i]
        batch = default_collate([item])
        raw_item = val_dataset_raw[i]
        return batch, raw_item, item

    patient2indices = val_dataset.list_indices_by_patient_ids()

    patients = patient_ids or sorted(list(patient2indices.keys()))

    os.makedirs(output_dir, exist_ok=True)

    for i, patient in enumerate(patients):
        if max_num and i >= max_num:
            break
        print(f"Processing patient {patient}")

        height = 3
        indices = patient2indices[patient]

        heatmap_fn = partial(
            show_heatmap,
            apply_prostate_mask=apply_prostate_mask,
            style=style,
            start_alpha=start_alpha,
            alpha_cutoff=alpha_cutoff,
            stop_alpha=stop_alpha,
            make_title=not no_title,
        )

        if mode == "one_image_per_core":
            for i in range(len(indices)):
                fig, ax = plt.subplots(1, 2, figsize=(8, height))
                batch, raw_item, item = get_batch(indices[i])
                dataset = "optimum"
                core_id = raw_item["info"]["cine_id"]
                if core_ids is not None and core_id not in core_ids:
                    continue

                heatmap_fn(
                    batch,
                    raw_item,
                    item,
                    model,
                    ax,
                    dataset,
                )
                fig.tight_layout()

                format_ext = format if format.startswith('.') else f".{format}"
                plt.savefig(os.path.join(output_dir, f"{core_id}{format_ext}"), dpi=dpi)
        else:
            fig, ax = plt.subplots(len(indices), 2, figsize=(8, height * len(indices)))
            for i in range(len(indices)):
                batch, raw_item, item = get_batch(indices[i])
                dataset = "optimum"
                heatmap_fn(
                    batch,
                    raw_item,
                    item,
                    model,
                    ax[i],
                    dataset,
                )
                fig.tight_layout()

            format_ext = format if format.startswith('.') else f".{format}"
            plt.savefig(os.path.join(output_dir, f"{patient}{format_ext}"), dpi=dpi)
            plt.close()


def show_heatmap(
    batch,
    raw_item,
    item,
    model,
    ax,
    dataset,
    apply_prostate_mask=False,
    style=None,
    start_alpha=0,
    stop_alpha=0.7,
    alpha_cutoff=0.1,
    make_title=True,
):

    out = model.forward(batch, include_postprocessed_heatmaps=True)

    if "image_level_classification_outputs" in out:
        # compute cspca score
        pre_bin_score = (
            out["image_level_classification_outputs"][0].softmax(-1)[:, 1].item()
        )
        # hacky way to get model score
        model_score = pd.cut(
            [pre_bin_score], bins=PROSTNFOUND_PLUS_FINAL_BINS, labels=range(1, 6)
        ).tolist()[0]
    else:
        model_score = None

    heatmap = out["cancer_probs"][0][0].cpu().numpy()

    if dataset == "nct2013":
        raw_image = raw_item["bmode"]
        prostate_mask = raw_item["prostate_mask"]
        needle_mask = raw_item["needle_mask"]

        raw_image = np.flipud(raw_image)
        prostate_mask = np.flipud(prostate_mask)
        needle_mask = np.flipud(needle_mask)
        heatmap = np.flipud(heatmap)

    else:
        raw_image = np.array(raw_item["image"])
        prostate_mask = np.array(raw_item["microsegnet_prostate_mask"])
        needle_mask = np.array(raw_item["needle_mask"])
        heatmap = np.flipud(heatmap)

    # heatmap = skimage.transform.resize(heatmap, raw_image.shape)
    height_mm = raw_item.get("info", {}).get("heightMm", 28)
    width_mm = raw_item.get("info", {}).get("widthMm", 46.06)

    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    def cancer_prob_colormap(alpha_cutoff=0.1, start_alpha=0, stop_alpha=1):
        """
        Returns a colormap that fades in from transparent to jet-like colors.
        Values below `alpha_cutoff` will be fully transparent.
        """
        base = plt.cm.jet(np.linspace(0, 1, 256))

        # Create alpha channel: transparent below cutoff, then ramp to 1
        alpha = np.linspace(start_alpha, stop_alpha, 256)
        alpha[: int(256 * alpha_cutoff)] = 0  # Fully transparent low values
        base[:, -1] = alpha  # Set alpha channel

        return mcolors.ListedColormap(base, name="cancer_prob_jet_alpha")

    cancer_cmap = cancer_prob_colormap(
        alpha_cutoff=alpha_cutoff, start_alpha=start_alpha, stop_alpha=stop_alpha
    )

    # basic heatmap for display
    extent = [0, width_mm, 0, height_mm]
    extent_flipped = [0, width_mm, height_mm, 0]

    # fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].imshow(raw_image, cmap="gray", extent=extent)
    if style == "miccai":
        ax[0].contour(prostate_mask, extent=extent_flipped, colors="white", alpha=0.5)
    ax[0].set_title("Image")
    ax[0].set_axis_off()

    ax[1].imshow(raw_image, cmap="gray", extent=extent)

    if apply_prostate_mask:
        alpha = np.ones(heatmap.shape, dtype="float") * stop_alpha
        prostate_mask_resized = skimage.transform.resize(
            prostate_mask, heatmap.shape, preserve_range=True, order=0
        ).astype(bool)
        needle_mask_resized = skimage.transform.resize(
            needle_mask, heatmap.shape, preserve_range=True, order=0
        ).astype(bool)
        alpha[prostate_mask_resized & ~needle_mask_resized] = 0.4
        alpha[~prostate_mask_resized] = 0

    else:
        alpha = None

    ax[1].contour(needle_mask, extent=extent_flipped, colors="white", alpha=0.5)
    if apply_prostate_mask:
        ax[1].contour(prostate_mask, extent=extent_flipped, colors="black", alpha=0.5)
    
    heatmap_for_display = cancer_cmap((heatmap * 255).astype(np.uint8))
    heatmap_for_display[..., -1] = alpha if alpha is not None else heatmap_for_display[..., -1]
    artist = ax[1].imshow(
        heatmap_for_display, extent=extent, vmin=0, vmax=1, cmap=cancer_cmap, alpha=alpha
    )
    # ax[1].set_title("Heatmap overlay")
    ax[1].set_axis_off()
    # plt.contour(prostate_mask, extent=extent_flipped)

    if item["label"] == 0:
        title = f"{item['core_id']} | Benign"
    else:
        title = f"{item['core_id']} - {item['pct_cancer']} % cancer - Grade Group {item['grade_group']}"
    
    if title:
        ax[0].set_title(title)

    title = ""
    if dataset == "optimum":
        pri_mus = raw_item["info"]["PRI-MUS"]
        pi_rads = raw_item["info"]["PI-RADS"]
        try:
            title += f"PRI-MUS {int(pri_mus)}"
        except:
            pass
        if not np.isnan(pi_rads) and pi_rads is not None:
            try:
                title += f" | PI-RADS {int(pi_rads)}"
            except:
                pass
        if model_score is not None:
            title += f" | Model {model_score}"
        ax[1].set_title(title)

    if not make_title:
        ax[1].set_title("")
        ax[0].set_title("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--apply_prostate_mask", action="store_true")
    parser.add_argument(
        "--patient_ids",
        nargs="+",
        default=None,
        help="If specified, only export for these patients.",
    )
    parser.add_argument(
        "--core_ids",
        nargs="+",
        help="if specified, only generate heatmaps for these patients.",
        default=None,
    )
    parser.add_argument("--mode", choices=(None, "one_image_per_core"), default=None)
    parser.add_argument("--style", choices=(None, "miccai"))
    parser.add_argument("--max_num", type=int, default=None)
    parser.add_argument("--start_alpha", type=float, default=0)
    parser.add_argument("--stop_alpha", type=float, default=1)
    parser.add_argument("--alpha_cutoff", type=float, default=0.1)
    parser.add_argument(
        "--dataset_cfg_file", help="Path to dataset config file (yaml).", default=None
    )
    parser.add_argument(
        "--no_title", action="store_true", help="If set, do not show titles on heatmaps."
    )
    parser.add_argument(
        "--format", type=str, default='png', help="Output format for saved heatmaps"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved heatmaps."
    )
    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        apply_prostate_mask=args.apply_prostate_mask,
        patient_ids=args.patient_ids,
        core_ids=args.core_ids,
        mode=args.mode,
        style=args.style,
        max_num=args.max_num,
        start_alpha=args.start_alpha,
        stop_alpha=args.stop_alpha,
        alpha_cutoff=args.alpha_cutoff,
        dataset_cfg_file=args.dataset_cfg_file,
        no_title=args.no_title,
        format=args.format,
        dpi=args.dpi,
    )
