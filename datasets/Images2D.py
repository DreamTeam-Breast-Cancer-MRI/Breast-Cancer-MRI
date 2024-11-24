import os
from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_positive, img_negatives, transform):
        self.img_dir_positives = img_positive
        self.images_positves = sorted(os.listdir(img_positive))

        self.negatives_offset = len(self.images_positves)

        self.img_dir_negatives = img_negatives
        self.images_negatives = sorted(os.listdir(img_negatives))

        self.transform = transform

        self.targets = [1 for _ in range(len(self.images_positves))]
        self.targets.extend([0 for _ in range(len(self.images_negatives))])

    def __len__(self):
        return len(self.images_positves) + len(self.images_negatives)

    def __getitem__(self, idx):
        if idx < self.negatives_offset:
            img_path = os.path.join(self.img_dir_positives, self.images_positves[idx])
            image = read_image(img_path)
            return self.transform(image), 1
        else:
            img_path = os.path.join(
                self.img_dir_negatives,
                self.images_negatives[idx - self.negatives_offset],
            )
            image = read_image(img_path)
            return self.transform(image), 0
