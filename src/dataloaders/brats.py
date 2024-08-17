from pathlib import Path
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

# resample image - sitk.BSpline as interpolator
def resample_image(img, size=[256, 256, 155]):
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # compute new spacing
    new_spacing2 = (img.GetSize()[2] / size[2]) * img.GetSpacing()[2]
    new_spacing1 = (img.GetSize()[1] / size[1]) * img.GetSpacing()[1]
    new_spacing0 = (img.GetSize()[0] / size[0]) * img.GetSpacing()[0]
    spacing = (new_spacing0, new_spacing1, new_spacing2)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return sitk.Resample(
        img, size, identity, sitk.sitkBSpline, origin, spacing, direction
    )

# resample label - sitk.sitkNearestNeighbor as interpolator
def resample_label(img, size=[256, 256, 155]):
    # print("og spacing", img.GetSpacing())
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # compute new spacing
    new_spacing2 = (img.GetSize()[2] / size[2]) * img.GetSpacing()[2]
    new_spacing1 = (img.GetSize()[1] / size[1]) * img.GetSpacing()[1]
    new_spacing0 = (img.GetSize()[0] / size[0]) * img.GetSpacing()[0]
    spacing = (new_spacing0, new_spacing1, new_spacing2)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return sitk.Resample(
        img, size, identity, sitk.sitkNearestNeighbor, origin, spacing, direction
    )

class BRATS(Dataset):
    def __init__(
        self,
        path,
        mode="train",
        subset=0.05,
        images=["flair", "t1", "t1ce", "t2"],
        size=[64, 64, 155],
        channels=4
    ):
        if path is None:
            RuntimeWarning("Dataset path is not set!")
        self.Path = Path(path)
        self.curPath = Path(path)
        self.imgmodes = images
        self.hgg = list(self.Path.glob(f"HGG/*{size[0]}.npy"))
        self.lgg = list(self.Path.glob(f"LGG/*{size[0]}.npy"))

        self.data = []

        subset_hgg = int(self.hgg.__len__() * subset)
        subset_lgg = int(self.lgg.__len__() * subset)
        subset_hgg_val = int(self.hgg.__len__() * min(subset + 0.1, 1.0))
        subset_lgg_val = int(self.lgg.__len__() * min(subset + 0.1, 1.0))

        if mode == "train":
            self.data = self.hgg[:subset_hgg] + self.lgg[:subset_lgg]
        elif mode == "train-hgg":
            self.data = self.hgg[:subset_hgg]
        elif mode == "train-lgg":
            self.data = self.lgg[:subset_lgg]
        elif mode == "test":
            self.data = self.hgg[subset_hgg_val:] + self.lgg[subset_lgg_val:]
        elif mode == "test-hgg":
            self.data = self.hgg[subset_hgg_val:]
        elif mode == "test-lgg":
            self.data = self.lgg[subset_lgg_val:]
        elif mode == "val":
            self.data = (self.hgg[subset_hgg:subset_hgg_val] + self.lgg[subset_lgg:subset_lgg_val])
        elif mode == "convert":
            for folder in ["HGG", "LGG"]:
                self.flairimgs = list(self.Path.glob(folder + "/*/*flair.nii.gz"))
                self.t1imgs = list(self.Path.glob(folder + "/*/*t1.nii.gz"))
                self.t1ceimgs = list(self.Path.glob(folder + "/*/*t1ce.nii.gz"))
                self.t2imgs = list(self.Path.glob(folder + "/*/*t2.nii.gz"))
                self.labels = list(self.Path.glob(folder + "/*/*seg.nii.gz"))
                assert len(self.flairimgs) == len(self.labels)

                self.img = [self.flairimgs, self.t1imgs, self.t1ceimgs, self.t2imgs]
                self.mask = self.labels
                for index in range(0, len(self.labels)):
                    img = []
                    savefile = self.img[0][index].parent.absolute().__str__()
                    for i in range(len(self.img)):
                        tmp = sitk.ReadImage(self.img[i][index].absolute().__str__())
                        tmp = resample_image(tmp, size=size)
                        img.append(sitk.GetArrayFromImage(tmp))

                    label = sitk.ReadImage(self.labels[index].absolute().__str__())
                    label = resample_label(label, size=size)

                    img.append(sitk.GetArrayFromImage(label))
                    img = np.stack(img)
                    self.data.append(img)
                    np.save(f"{savefile}_{size[0]}", img)
        else:
            print("Please use 'train','test','val' or 'convert' as mode")
            return False

    def __len__(self):
        return self.data.__len__() * 155

    def __getitem__(self, item):
        data = np.load(self.data[item // 155].absolute().__str__())
        img = torch.from_numpy(data[:4, item % 155])# [4, 64, 64]
        mask = torch.from_numpy(data[-1, item % 155])# [1, 64, 64]
        mask[4 == mask] = 3
        img = img.float()
        mask = mask.float()
        mask[mask > 0] = 1.0
        mask = torch.unsqueeze(mask, dim=0)
        return img, mask, [mask]

if __name__ == "__main__":

    d = BRATS(
        path='../../../../special-course/data/MICCAI_BraTS_2018_Data_Training',
        subset=0.6,
        mode="convert",
    )
    print(len(d))