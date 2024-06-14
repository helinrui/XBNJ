import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class XBNJ(Dataset):
    def __init__(self, root_dir, mode="train", args=None, transform=None):
        assert mode in ['train', 'val'], "Invalid mode"
        self.root_dir = root_dir
        self.mode = mode
        self.args = args
        self.img_folder = os.path.join(root_dir, 'imgall23diff22', mode)
        self.labels_file = os.path.join(root_dir, f'labelsdiff_{mode}.txt')  # 根据model值动态加载不同的标签文件
        # self.class_mapping = {'增生性息肉': 0, '腺瘤': 1, 'ssl': 2, 'Ca': 3, '高质量': 4, '低质量': 5}
        self.data = self.load_data()
        self.transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop((224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 0.264, 0.178, 0.159
            ])
        }

    def load_data(self):
        with open(self.labels_file, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            parts = line.split()
            img_path = os.path.join(self.img_folder, parts[0])
            label = [int(parts[i]) for i in range(1, len(parts))]
            data.append((img_path, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform[self.mode](img)  # 应用变换
        label_tensor = torch.tensor(label[0])

        return img, label_tensor


if __name__ == "__main__":
   pass