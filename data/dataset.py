import os
import random
from typing import Optional
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(self, root_dir: str, num_classes: int = 53, img_exts=(".jpg", ".jpeg", ".png", ".bmp"),transform=None, augment=True, img_size=128):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.img_exts = img_exts
        self.img_size = img_size
        self.target_size = (img_size, img_size)
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.augment_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]) if augment else None

        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        self.custom_transform = transform

        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        if os.path.exists(root_dir):
            self._load_samples()
        else:
            raise ValueError(f"Folder {root_dir} not exist")
        self._print_class_info()

    def _load_samples(self):
        class_folders = []
        for item in sorted(os.listdir(self.root_dir)):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                class_folders.append(item)

        for class_idx, class_folder in enumerate(class_folders):
            self.class_names.append(class_folder)
            self.class_to_idx[class_folder] = class_idx

            class_path = os.path.join(self.root_dir, class_folder)
            class_images = []

            for file in os.listdir(class_path): 
                if any(file.lower().endswith(ext) for ext in self.img_exts):
                    img_path = os.path.join(class_path, file)
                    if self._is_valid_image(img_path):
                        class_images.append(img_path)

            for img_path in sorted(class_images):
                self.samples.append((img_path, class_idx))

            print(f"Class {class_folder}: {len(class_images)} item")

    def _is_valid_image(self, img_path: str) -> bool:
        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_tensor = transforms.ToTensor()(img)
                if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                    print(f"Invalid tensor value: {img_path}")
                    return False

            return True
        except Exception as e:
            print(f"Invalid image {img_path}: {str(e)}")
            return False

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        if not self.augment or random.random() < 0.3:
            return img
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.5, 1.5))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            print(f"Index {idx} is invalid")
            return self._get_default_item()

        img_path, class_label = self.samples[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            img = self._enhance_image(img)

            if self.custom_transform:
                img = self.custom_transform(img)
            else:
                if self.augment and self.augment_transforms:
                    img = self.augment_transforms(img)

                img = self.base_transform(img)

                if img.shape[0] == 3: 
                    img = transforms.functional.rgb_to_grayscale(img, num_output_channels=1)

                img = self.normalize(img)

            if img.shape != (1, self.img_size, self.img_size):
                print(f"Error size: {img_path}: {img.shape}")
                return self._get_default_item()

            if torch.isnan(img).any() or torch.isinf(img).any():
                print(f"Invalid value {img_path}")
                return self._get_default_item()

            return img, torch.tensor(class_label, dtype=torch.long)

        except Exception as e:
            print(f"Upload image fail {img_path}: {str(e)}")
            return self._get_default_item()

    def _get_default_item(self):
        default_img = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        default_label = torch.tensor(0, dtype=torch.long)
        return default_img, default_label

    def get_class_name(self, class_idx):
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"unknown_class_{class_idx}"

    def get_class_distribution(self):
        distribution = {}
        for _, class_label in self.samples:
            class_name = self.get_class_name(class_label)
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def _print_class_info(self):
        distribution = self.get_class_distribution()
        total_samples = len(self.samples)

        print("-" * 50)
        for class_name, count in distribution.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count} item ({percentage:.1f}%)")
        print("-" * 50)

        max_count = max(distribution.values())
        min_count = min(distribution.values())
        if max_count / min_count > 3:
            print("Unbalance dataset.")

    def get_class_weights(self):
        distribution = self.get_class_distribution()
        total_samples = len(self.samples)

        weights = []
        for i in range(len(self.class_names)):
            class_name = self.get_class_name(i)
            class_count = distribution.get(class_name, 1)
            weight = total_samples / (len(self.class_names) * class_count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def improved_collate_fn(batch):
    valid_items = []

    for idx, item in enumerate(batch):
        if item is None:
            print(f"Item {idx} is None")
            continue
        img, target = item
        if img.shape != (1, 128, 128):
            print(f"Item {idx}: Error shape {img.shape}")
            continue
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"Item {idx}: NaN/Inf")
            continue
        if not (0 <= target < 53):
            print(f"Item {idx}: Invalid label {target}")
            continue
        valid_items.append(item)

    if len(valid_items) == 0:
        print("All item invalid")
        return torch.empty(0, 1, 128, 128), torch.empty(0, dtype=torch.long)

    imgs = torch.stack([item[0] for item in valid_items], dim=0)
    targets = torch.stack([item[1] for item in valid_items], dim=0)
    print(f"Batch shape valid: {len(valid_items)}")
    return imgs, targets
