import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class OCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 8192)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()

        return image, label
