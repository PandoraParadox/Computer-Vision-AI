import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassificationLoss(nn.Module):
    def __init__(self, num_classes=53, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
        self.loss_history = []
        self.max_history = 50
        
    def forward(self, outputs, targets):
        device = outputs.device
        batch_size = outputs.size(0)
        
        outputs, targets = self._validate_inputs(outputs, targets)
        
        try:
            cls_loss = self.cross_entropy(outputs, targets)
            cls_loss = torch.clamp(cls_loss, 0.0, 10.0)
        except Exception as e:
            print(f"Lỗi tính loss: {e}")
            cls_loss = self._fallback_loss_calculation(outputs, targets)
        
        self._update_loss_history(cls_loss.item())
        
        total_loss = cls_loss
        box_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)
        
        return total_loss, box_loss, conf_loss, cls_loss
    
    def _validate_inputs(self, outputs, targets):
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs = torch.clamp(outputs, -8.0, 8.0)
        
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        targets = torch.clamp(targets, 0, self.num_classes - 1)
        
        return outputs, targets
    
    def _fallback_loss_calculation(self, outputs, targets):
        try:
            outputs_safe = torch.clamp(outputs, -5, 5)
            log_probs = F.log_softmax(outputs_safe, dim=1)
            nll_loss = F.nll_loss(log_probs, targets, reduction='mean')
            return torch.clamp(nll_loss, 0.0, 8.0)
        except:
            print("Sử dụng fallback loss cuối cùng")
            return torch.tensor(2.0, device=outputs.device, requires_grad=True)
    
    def _update_loss_history(self, loss_value):
        self.loss_history.append(loss_value)
        if len(self.loss_history) > self.max_history:
            self.loss_history.pop(0)
    
    def calculate_accuracy(self, outputs, targets):
        try:
            with torch.no_grad():
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)
                predictions = torch.argmax(outputs, dim=1)
                
                if targets.dim() > 1:
                    targets = targets.squeeze()
                
                correct = (predictions == targets).float().sum()
                total = targets.size(0)
                accuracy = correct / total
                
                return torch.clamp(accuracy, 0.0, 1.0).item()
        except:
            return 0.0
    
    def get_loss_stats(self):
        if len(self.loss_history) < 2:
            return {}
        
        return {
            'mean': np.mean(self.loss_history),
            'std': np.std(self.loss_history),
            'min': np.min(self.loss_history),
            'max': np.max(self.loss_history),
            'recent': self.loss_history[-1] if self.loss_history else 0.0
        }