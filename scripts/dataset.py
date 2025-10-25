import os
from torch.utils.data import Dataset
import cv2
from PIL import Image

class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, folders, transform=None):
        self.root_dir = root_dir
        self.folders = folders
        self.transform = transform
        self.files = []

        # build list of all files
        for s in self.folders:
            path = os.path.join(root_dir, s)
            for f in os.listdir(path):
                f_path = os.path.join(path, f)
                for x in os.listdir(f_path):
                    self.files.append(os.path.join(f_path, x))

        # class names from folder structure
        class_names = sorted({os.path.basename(os.path.dirname(fp)) for fp in self.files})
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        class_name = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[class_name]

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)

        return img, label
