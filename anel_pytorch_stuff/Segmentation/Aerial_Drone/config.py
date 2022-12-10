from torch import nn

dataset_path = "semantic-drone-dataset"

config = {
    "num_clases": 24,
    "dataset_path": "semantic-drone-dataset",
    "img_path": dataset_path + "/dataset/semantic_drone_dataset/original_images/",
    "mask_path": dataset_path + "/RGB_color_image_masks/RGB_color_image_masks/",
    "batchsize": 4,
    "label_file_path": "semantic-drone-dataset/class_dict_seg.csv",
    "criterion": nn.CrossEntropyLoss(),
    "epochs": 2,
    "target_height": 512,
    "target_width": 512,
}
