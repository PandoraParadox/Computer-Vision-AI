import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 53):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.clamp(x, -3, 3)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.features(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        x = torch.clamp(x, -5, 5)
        
        return x

class ResidualBlock(nn.Module):
    """Khối residual để cải thiện gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        residual = self.skip_connection(residual)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNetClassifier(nn.Module):
    """Mô hình ResNet để gradient flow tốt hơn"""
    def __init__(self, num_classes: int = 53, dropout_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.clamp(x, -3, 3)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        x = torch.clamp(x, -8, 8)
        
        return x

def create_stable_model(model_type='simple', num_classes=53, dropout_rate=0.2):
    if model_type == 'simple':
        return ClassificationModel(num_classes)
    elif model_type == 'resnet':
        return ResNetClassifier(num_classes, dropout_rate)
    elif model_type == 'stable':
        return ClassificationModel(num_classes)
    else:
        raise ValueError(f"Loại mô hình không xác định: {model_type}")

def test_model_stability(model, device='cpu', num_tests=10):
    model.eval()
    model.to(device)
    
    print(f"Kiểm tra độ ổn định mô hình trên {device}...")
    
    gradient_norms = []
    output_ranges = []
    
    for i in range(num_tests):
        x = torch.randn(4, 1, 128, 128, device=device) * 0.5  # Sử dụng kích thước 128
        target = torch.randint(0, model.num_classes, (4,), device=device)
        
        model.train()
        x.requires_grad_(True)
        
        output = model(x)
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        gradient_norms.append(total_norm)
        output_ranges.append((output.min().item(), output.max().item()))
        
        print(f"Test {i+1}: Loss={loss.item():.4f}, GradNorm={total_norm:.4f}, "
              f"OutputRange=[{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print(f"\nTóm tắt độ ổn định:")
    print(f"Norm gradient trung bình: {np.mean(gradient_norms):.4f}")
    print(f"Norm gradient tối đa: {np.max(gradient_norms):.4f}")
    print(f"Độ lệch chuẩn gradient: {np.std(gradient_norms):.4f}")
    
    if np.max(gradient_norms) > 50:
        print("⚠️ CẢNH BÁO: Phát hiện gradient nổ!")
    elif np.max(gradient_norms) < 0.01:
        print("⚠️ CẢNH BÁO: Phát hiện gradient biến mất!")
    else:
        print("✅ Mô hình ổn định!")
    
    return gradient_norms, output_ranges

if __name__ == "__main__":
    import numpy as np
    for model_type in ['simple', 'resnet', 'stable']:
        print(f"\n{'='*50}")
        print(f"Kiểm tra mô hình {model_type}")
        print(f"{'='*50}")
        model = create_stable_model(model_type, num_classes=53)
        test_model_stability(model, device='cpu', num_tests=5)