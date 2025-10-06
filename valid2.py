
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from models.backbone import ResNetClassifier  


def load_model(model_path, num_classes=3, device='cpu'):
    model = ResNetClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def predict_single_image_top3(model, image_path, device, class_names, img_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: transforms.functional.rgb_to_grayscale(x, num_output_channels=1) 
            if x.shape[0] == 3 else x
        ),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0] 
        top3_probs, top3_indices = torch.topk(probabilities, k=3)

    print(f"\nẢnh: {os.path.basename(image_path)}")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        class_name = class_names[idx.item()] if idx.item() < len(class_names) else f"unknown_{idx.item()}"
        print(f"  {i+1}. {class_name} (class {idx.item()}): {prob.item()*100:.2f}%")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "weights/best_model22.pth"  
    image_path = "data/test/celldark2.jpg"
    train_dir = "data/train"

    class_names = sorted([
        d for d in os.listdir(train_dir) 
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    model = load_model(model_path, num_classes=3, device=device)
    predict_single_image_top3(model, image_path, device, class_names, img_size=128)
