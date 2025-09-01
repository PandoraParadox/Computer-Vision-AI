import torch
from models.backbone import MyDetector

checkpoint_path = 'weights/model_epoch_7.pth'
onnx_model_path = 'weights/model_epoch_7.onnx'

model = MyDetector(num_classes=6, S=7, num_anchors=2)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
print(f"Đã lưu mô hình ONNX tại: {onnx_model_path}")

import onnx
model_onnx = onnx.load(onnx_model_path)
onnx.checker.check_model(model_onnx)
print("Mô hình ONNX hợp lệ!")