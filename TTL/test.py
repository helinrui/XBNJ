import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import models
from collections import OrderedDict
import sys
sys.path.extend("./utils/")
from utils.models import resnet50
from utils.settings import parse_opts

class XBNJ(Dataset):
    def __init__(self, root_dir, mode="train", args=None, transform=None):
        assert mode in ['train', 'val'], "Invalid mode"
        self.root_dir = root_dir
        self.mode = mode
        self.args = args
        self.img_folder = os.path.join(root_dir, 'imgfinall23', mode)
        self.labels_file = os.path.join(root_dir, f'labels_{mode}.txt')
        self.data = self.load_data()
        self.transform = transform

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
        if self.transform:
            img = self.transform(img)
        label_tensor = torch.tensor(label[0])

        return img, label_tensor

if __name__ == "__main__":
    # 加载模型
    args = parse_opts()
    root_dir = r'/home/helinrui/slns/TTL/data'
    model_path = "/home/helinrui/slns/TTL/checkpoint/XBNJ/resnet50_imagenet_-1_10/best.pt"
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')  # 检查是否有GPU可用
    checkpoint = torch.load(model_path, map_location=device)
    # 创建一个与你之前训练模型相同的模型实例
    model = resnet50()
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['model_state_dict'].items():
    #     name = k.replace("module.", "")  # 去掉额外的前缀 "module."
    #     new_state_dict[name] = v
    # 加载状态字典到模型中
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = XBNJ(root_dir=root_dir,mode="val", args=args, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=64,  shuffle=False)

    mismatched_images = set()

    # 测试模型
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 将批次中每个样本的预测标签逐一记录
            for j in range(len(labels)):
                pred_label = predicted[j].item()
                true_label = labels[j].item()
                if pred_label != true_label:
                    img_name = os.path.basename(test_dataset.data[i * test_loader.batch_size + j][0])
                    mismatched_images.add((img_name, pred_label, true_label))

    # 输出预测标签和实际标签不一致的图片名字
    for idx, (img_name, predicted_label, true_label) in enumerate(mismatched_images):
        print(f"Image{idx}:", f"Image: {img_name}, Predicted Label: {predicted_label}, True Label: {true_label}")
