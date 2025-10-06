import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import csv
from tqdm import tqdm

from models.backbone import ClassificationModel

class TestDataset:
    def __init__(self, test_dir, img_size=128, img_exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.test_dir = test_dir
        self.img_size = img_size
        self.img_exts = img_exts
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: transforms.functional.rgb_to_grayscale(x, num_output_channels=1) if x.shape[0] == 3 else x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_paths = []
        self._load_image_paths()
    
    def _load_image_paths(self):
        if not os.path.exists(self.test_dir):
            raise ValueError(f"Thư mục test {self.test_dir} không tồn tại")
        for file in sorted(os.listdir(self.test_dir)):
            if any(file.lower().endswith(ext) for ext in self.img_exts):
                img_path = os.path.join(self.test_dir, file)
                if self._is_valid_image(img_path):
                    self.image_paths.append(img_path)
        print(f"Tìm thấy {len(self.image_paths)} hình ảnh hợp lệ trong thư mục test")
    
    def _is_valid_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"Ảnh không hợp lệ {img_path}: {str(e)}")
            return False
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            if img_tensor.shape != (1, self.img_size, self.img_size):
                return None, img_path
            if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                return None, img_path
            return img_tensor, img_path
        except Exception as e:
            print(f"Lỗi khi load ảnh {img_path}: {str(e)}")
            return None, img_path

def load_model(model_path, num_classes=3, device='cpu'):
    print(f"Đang load model từ {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ClassificationModel(num_classes=num_classes)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model đã được load thành công!")
    return model

def load_class_names(train_dir):
    class_names = []
    if os.path.exists(train_dir):
        for item in sorted(os.listdir(train_dir)):
            item_path = os.path.join(train_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
    print(f"Tìm thấy {len(class_names)} class")
    return class_names

def test_collate_fn(batch):
    valid_items = []
    paths = []
    for img, path in batch:
        if img is not None:
            valid_items.append(img)
        paths.append(path)
    if len(valid_items) == 0:
        return torch.empty(0, 1, 128, 128), paths
    imgs = torch.stack(valid_items, dim=0)
    return imgs, paths

def predict_batch(model, test_loader, device, class_names, output_file="predictions.csv"):
    print("Bắt đầu dự đoán...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (imgs, paths) in enumerate(tqdm(test_loader, desc="Predicting")):
            if len(imgs) == 0:
                for path in paths:
                    predictions.append({
                        'image_path': os.path.basename(path),
                        'predicted_class': -1,
                        'class_name': 'invalid_image',
                        'confidence': 0.0
                    })
                continue
            imgs = imgs.to(device)
            outputs = model(imgs)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_classes = torch.max(probabilities, 1)
            for i, path in enumerate(paths):
                if i < len(predicted_classes):
                    pred_class = predicted_classes[i].item()
                    confidence = confidences[i].item()
                    if 0 <= pred_class < len(class_names):
                        class_name = class_names[pred_class]
                    else:
                        class_name = f"unknown_class_{pred_class}"
                    predictions.append({
                        'image_path': os.path.basename(path),
                        'predicted_class': pred_class,
                        'class_name': class_name,
                        'confidence': confidence
                    })
                else:
                    predictions.append({
                        'image_path': os.path.basename(path),
                        'predicted_class': -1,
                        'class_name': 'prediction_failed',
                        'confidence': 0.0
                    })
    print(f"Lưu kết quả vào {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'predicted_class', 'class_name', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)
    print(f"Hoàn thành! Đã dự đoán {len(predictions)} ảnh.")
    valid_predictions = [p for p in predictions if p['predicted_class'] >= 0]
    if valid_predictions:
        avg_confidence = np.mean([p['confidence'] for p in valid_predictions])
        print(f"Độ tin cậy trung bình: {avg_confidence:.4f}")
        class_counts = {}
        for pred in valid_predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print("\nTop 5 class được dự đoán nhiều nhất:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {class_name}: {count} ảnh")
    return predictions

def main():
    test_dir = "data/test"
    train_dir = "data/train"
    model_path = "weights/best_model.pth"
    output_file = "predictions.csv"
    batch_size = 8
    img_size = 128
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    class_names = load_class_names(train_dir)
    if len(class_names) != num_classes:
        print(f"Cảnh báo: Số class tìm thấy ({len(class_names)}) khác với num_classes ({num_classes})")
    model = load_model(model_path, num_classes, device)
    test_dataset = TestDataset(test_dir, img_size=img_size)
    if len(test_dataset) == 0:
        print("Không tìm thấy ảnh hợp lệ nào trong thư mục test!")
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=0
    )
    predict_batch(model, test_loader, device, class_names, output_file)

if __name__ == "__main__":
    main()
