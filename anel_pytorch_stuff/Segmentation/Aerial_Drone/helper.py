import albumentations as A
import numpy as np


# Convert 3D rgb color to 1D mask idx
def rgb_to_mask(source_rgb_mask, classname_to_rgb_mapping, class_names):
    source_rgb_mask = np.array(source_rgb_mask)
    height, width, channels = source_rgb_mask.shape
    cls_mask = np.ones(source_rgb_mask.shape, dtype=np.uint8)

    for classname in classname_to_rgb_mapping:
        current_cls_mask = source_rgb_mask == classname_to_rgb_mapping[classname]
        current_cls_mask = current_cls_mask.all(axis=2)
        cls_mask[current_cls_mask] = class_names.index(classname)

    cls_mask = cls_mask[:, :, 0]
    cls_mask = np.reshape(cls_mask, (height, width, 1))

    return cls_mask


# Convert 1D msk idx to 3D rgb color
def convert_mask_to_rgb(predicted_mask, classname_to_rgb_mapping, class_names):
    rgb_img = np.stack([predicted_mask, predicted_mask, predicted_mask], axis=2).astype(
        int
    )
    rgb_img = rgb_img.squeeze()
    height, width, channels = predicted_mask.shape
    rgb_img = rgb_img.reshape([height * width, 3])

    for pixel_idx, pixel_rgb_vals in enumerate(rgb_img):
        rgb_img[pixel_idx] = classname_to_rgb_mapping[class_names[pixel_rgb_vals[0]]]

    rgb_img = rgb_img.reshape([height, width, 3])

    return rgb_img


def get_transforms(target_height, target_width):
    training_transform = A.Compose(
        [
            A.Resize(target_height, target_width, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.OpticalDistortion(distort_limit=0.25, shift_limit=0.25, p=1),
        ]
    )

    inference_transform = A.Compose([A.Resize(target_height, target_width, p=1)])

    return training_transform, inference_transform


# Function creates dict from dataframe
def label_df_to_dict(label_df):
    return {
        row["name"]: [row[" r"], row[" g"], row[" b"]]
        for index, row in label_df.iterrows()
    }
