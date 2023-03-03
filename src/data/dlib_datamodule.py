from typing import Any, Dict, Optional, Tuple

import PIL.Image
from PIL import ImageOps
import numpy as np
import torch
import torchvision.io
from PIL.Image import Image
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.io import ImageReadMode
from torchvision.transforms import transforms, InterpolationMode
import os
import cv2


def read_pts(path):
    return np.loadtxt(path, comments=("version:", "n_points:", "{", "}"))


class DlibDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "F:/project/Testing/Filter/data/300w",
            train_val_test_split: list[int, int, int] = [560, 20, 20],
            batch_size: int = 10,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # allows access to init params
        self.save_hyperparameters(logger=False)

        # data types
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        # data transformation
        self.transforms = transforms.Compose([
            transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize((86.43806780133929 / 256, 92.93189074457908 / 256, 106.79691941034226 / 256,),
            #                      (75.54991500222036 / 256, 76.3969277879787 / 256, 80.96140965619875 / 256,))
        ]
        )
        self.out_transform = transforms.Compose([

        ])
    @property
    def num_classes(self):
        return 68

    """ Load data, set variables, split and akamudabra
    """

    def setup(self, stage: Optional[str] = None):
        inputs = torch.empty(0, 3, 224, 224)
        outputs = torch.empty(0, 68, 2)
        if not self.data_train and not self.data_val and not self.data_test:
            path = self.hparams.data_dir

            for dir in os.listdir(path):
                sub_path = path + '/' + dir
                if not os.path.isfile(sub_path):
                    for sub_dir in sorted(os.listdir(sub_path)):
                        data_path = sub_path + "/" + sub_dir
                        try:
                            img = PIL.Image.open(data_path).convert("RGB")
                            print("Reading image: {}".format(data_path))
                            ratio = [img.width / 224, img.height / 224]
                            inputs = torch.cat([inputs,
                                                self.transforms(img).unsqueeze(0)], 0)
                            txt_path = data_path[0:len(data_path) - 4] + ".pts"
                            outputs = torch.cat([outputs,
                                                torch.tensor(read_pts(txt_path) / ratio).unsqueeze(0)], 0)
                        except IOError:
                            pass
        print(inputs.size(), outputs.size())
        data = torch.utils.data.TensorDataset(inputs, outputs)

        self.data_train, self.data_val, self.data_test = random_split(
            dataset=data,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


def main():
    data = DlibDataModule()
    data.setup()
    loader = data.test_dataloader()
    fig = plt.figure(figsize=(5, 2))
    i = 1
    for test_img, test_label in loader:
        for i in range(1, 11):
            fig.add_subplot(5, 2, i)
            plt.axis("off")
            plt.imshow(test_img[i - 1].permute(1, 2, 0))
            plt.scatter(test_label[i - 1, :, 0], test_label[i - 1, :, 1], color="r", s=2)
    plt.show()


if __name__ == "__main__":
    main()
