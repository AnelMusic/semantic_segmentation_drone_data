from glob import glob

import torch

from config import config
from dataset import get_dataloaders
from model import get_model, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    train_loader, test_loader = get_dataloaders(
        config["img_path"],
        config["mask_path"],
        config["label_file_path"],
        config["batchsize"],
        config["target_height"],
        config["target_width"],
    )
    model = get_model(config["num_clases"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = train_model(model, train_loader, test_loader, config, optimizer, device)


if __name__ == "__main__":
    main()
