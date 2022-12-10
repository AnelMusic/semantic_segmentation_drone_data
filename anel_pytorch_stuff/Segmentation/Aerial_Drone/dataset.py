from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image

from helper import get_transforms, label_df_to_dict, rgb_to_mask


class AerialDroneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_path,
        mask_path,
        training,
        classname_to_rgb_mapping,
        class_names,
        inference_transform,
        training_transform,
    ):
        super(AerialDroneDataset, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.training = training
        self.inference_transform = inference_transform
        self.training_transform = training_transform
        self.img_paths = sorted(glob(self.img_path + "*.jpg"))
        self.mask_paths = sorted(glob(self.mask_path + "*.png"))
        self.classname_to_rgb_mapping = classname_to_rgb_mapping
        self.class_names = class_names

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # needs 3c->1c conversion to

        image = np.array(image)
        mask = np.array(mask)
        mask = rgb_to_mask(mask, self.classname_to_rgb_mapping, self.class_names)

        if self.training == True:
            augmentations = self.training_transform(image=image, masks=[mask])
            image = augmentations["image"]
            mask = augmentations["masks"][0]
        else:
            augmentations = self.inference_transform(image=image, masks=[mask])
            image = augmentations["image"]
            mask = augmentations["masks"][0]

        # normalize img but not mask
        image = image / 255  # real mean and stdev yield better results
        height = image.shape[0]
        width = image.shape[1]
        image, mask = torch.tensor(image).float(), torch.tensor(mask, dtype=torch.int64)
        image = torch.reshape(image, (3, height, width))  # Reshape for Seg Models
        mask = torch.reshape(mask, (1, height, width))

        return image, mask

    def __len__(self):
        return len(self.img_paths)


def get_dataloaders(
    img_path, mask_path, label_file_path, batchsize, target_height, target_width
):
    labels_df = pd.read_csv(label_file_path)
    classname_to_rgb_mapping = label_df_to_dict(labels_df)
    class_names = list(classname_to_rgb_mapping.keys())
    training_transform, inference_transform = get_transforms(
        target_height, target_width
    )

    dataset = AerialDroneDataset(
        img_path,
        mask_path,
        True,
        classname_to_rgb_mapping,
        class_names,
        training_transform,
        inference_transform,
    )

    test_num = int(0.8 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True, num_workers=0
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False, num_workers=0
    )

    return train_dataloader, test_dataloader
