from re import I
from typing import Any, Dict, Literal
from medAI.datasets.nct2013.utils import load_or_create_resized_bmode_data
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from medAI.datasets.nct2013 import data_accessor
import numpy as np 
from medAI.utils.data.patch_extraction import PatchView
import torch
from PIL import Image
from typing_extensions import deprecated
import h5py


class BModePatchesDataset(Dataset):
    _metadata_table = data_accessor.get_metadata_table()

    def __init__(
        self,
        core_ids,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        needle_mask_threshold=0.6,
        prostate_mask_threshold=-1,
        transform=None,
        data_file='/h/pwilson/projects/medAI/data/nct2013/bmode.h5', 
        output_mode: Literal['single_patch', 'all_patches'] = 'single_patch'
    ):
        self.core_ids = core_ids
        self.output_mode = output_mode
        self.data_file = h5py.File(
            data_file, 'r'
        )
        im_size_mm = 28, 46.06
        im_size_px = self.get_image(core_ids[0]).shape
        self.patch_size_px = int(patch_size_mm[0] * im_size_px[0] / im_size_mm[0]), int(
            patch_size_mm[1] * im_size_px[1] / im_size_mm[1]
        )
        self.patch_stride_px = int(
            patch_stride_mm[0] * im_size_px[0] / im_size_mm[0]
        ), int(patch_stride_mm[1] * im_size_px[1] / im_size_mm[1])

        self._images = [self.get_image(core_id) for core_id in core_ids]
        self._prostate_masks = [
            data_accessor.get_prostate_mask(core_id) for core_id in core_ids
        ]
        self._needle_masks = [
            data_accessor.get_needle_mask(core_id) for core_id in core_ids
        ]

        self._patch_views = PatchView.build_collection_from_images_and_masks(
            image_list=self._images,
            window_size=self.patch_size_px,
            stride=self.patch_stride_px,
            align_to="topright",
            mask_lists=[self._prostate_masks, self._needle_masks],
            thresholds=[prostate_mask_threshold, needle_mask_threshold],
            image_shape=self.get_image(core_ids[0]).shape,
        )
        self._indices = []
        for i, pv in enumerate(self._patch_views):
            if self.output_mode == 'single_patch':
                self._indices.extend([(i, j) for j in range(len(pv))])
            else: 
                self._indices.append((i, range(len(pv))))

        self.transform = transform

    def get_image(self, core_id): 
        return self.data_file[core_id]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        
        if isinstance(idx, str): 
            assert self.output_mode == 'all_patches'
            i = self.core_ids.index(idx)
            j = None
            metadata = (
                self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
                .iloc[0]
                .to_dict()
            )
        else: 
            i, j = self._indices[idx]
            metadata = (
                self._metadata_table[self._metadata_table.core_id == self.core_ids[i]]
                .iloc[0]
                .to_dict()
            )

        pv: PatchView = self._patch_views[i]

        if self.output_mode == 'single_patch': 
            patch = pv[j]
            patch = patch.copy()
            postition = pv.positions[j]
            patch = Image.fromarray(patch).convert("L")
            data = {"patch": patch, **metadata, "position": postition}
    
        elif self.output_mode == 'all_patches': 
            patches = []
            positions = [] 
            for j in range(len(pv)): 
                patch = pv[j].copy()
                patch = Image.fromarray(patch).convert("L")
                patches.append(patch)
                positions.append(pv.positions[j])
            data = {
                "patch": patches,
                "position": positions, 
                **metadata
            }
        
        if self.transform is not None:
            data = self.transform(data)

        return data


class SelfNormalize(T.Transform): 
    _transformed_types = (torch.Tensor,)

    def _transform(self, inpt, params: Dict[str, Any]):
        inpt = inpt - inpt.mean()
        inpt = inpt / inpt.std()        
        return inpt


class Transform:
    def __init__(self, use_augmentations=False):
        self.use_augmentations = use_augmentations
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            ]
        )

    def __call__(self, item):
        patch = item["patch"]
        patch = T.Resize((128, 128))(patch)
        if self.use_augmentations:
            patch = self.augs(patch)
        patch = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True)],
        )(patch)
        patch = SelfNormalize()(patch)
        item["patch"] = patch
        return item


# class TransformV2(T.Compose): 
# 
#     _transformed_types = (Image.Image,)
# 
#     def __init__(self, use_augmentations=False): 
#         transform = []
#         transform.append(T.Resize((224, 224)))
#         if use_augmentations: 
#             transform.append(T.Compose(
#                 [
#                     T.RandomHorizontalFlip(),
#                     T.RandomVerticalFlip(),
#                     T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
#                 ]
#             ))
#         transform.append(
#             T.Compose([
#                 T.ToImage(), 
#                 T.ToDtype(torch.float32, scale=True), 
#                 SelfNormalize()
#             ])
#         )
#         super().__init__(transform)
    

class SSLTransform:
    def __init__(self): 
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                T.RandomResizedCrop((128, 128), scale=(0.3, 1.0), ratio=(0.75, 1.3333)),
                T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
            ]
        )

    def __call__(self, item):
        patch = item["patch"]
        patch = T.Resize((128, 128))(patch)
        p1 = self.augs(patch)
        p2 = self.augs(patch)
    
        p1 = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True), SelfNormalize()],
        )(p1)
        p2 = T.Compose(
            [T.ToImage(), T.ToDtype(torch.float32, scale=True), SelfNormalize()],
        )(p2)
        
        return p1, p2



if __name__ == "__main__": 
    from medAI.datasets.nct2013.cohort_selection import select_cohort

    train_cores, val_cores, test_cores = select_cohort(undersample_benign_ratio=6, involvement_threshold_pct=40)

    ds = RFPatchesDataset(
        train_cores, 
        data_type='bmode'
    )