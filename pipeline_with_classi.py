import cv2
import argparse
import json
import time
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.backbone import ClassificationModel


def load_model(model_path, num_classes=3, device='cpu'):
    model = ClassificationModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_single_image(model, crop_img, device, class_names, img_size=128, threshold=0.6):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: transforms.functional.rgb_to_grayscale(x, num_output_channels=1) if x.shape[0] == 3 else x
        ),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top1_prob, top1_idx = torch.max(probabilities, dim=0)

    if top1_prob.item() < threshold:
        class_name = "NaN"
    else:
        class_name = class_names[top1_idx.item()] if top1_idx.item() < len(class_names) else f"unknown_{top1_idx.item()}"

    return class_name, top1_prob.item()


class Pipeline:
    def __init__(self, config, model, device, class_names):
        self.min_box_area = config.get("min_box_area", 1500)
        self.max_distance = config.get("max_distance", 60)
        self.max_objects = config.get("max_objects", 10)
        self.movement_threshold = config.get("movement_threshold", 10)

        self.objects = {}
        self.next_id = 0

        self.model = model
        self.device = device
        self.class_names = class_names

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= self.min_box_area:
                new_boxes.append((x, y, w, h))

        new_boxes = sorted(new_boxes, key=lambda b: b[2]*b[3], reverse=True)[:self.max_objects]

        tracked = []
        updated_ids = set()

        for (bx, by, bw, bh) in new_boxes:
            cx, cy = bx + bw//2, by + bh//2

            matched_id, min_dist = None, float("inf")
            for oid, (ox, oy, ow, oh, last_seen, pcx, pcy, last_dir) in self.objects.items():
                dist = ((cx - pcx)**2 + (cy - pcy)**2) ** 0.5
                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    matched_id = oid

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
                prev_cx, prev_cy, prev_dir = cx, cy, None
            else:
                prev_cx, prev_cy, prev_dir = self.objects[matched_id][5], self.objects[matched_id][6], self.objects[matched_id][7]

            dx, dy = cx - prev_cx, cy - prev_cy
            move_dist = (dx**2 + dy**2) ** 0.5

            if move_dist > self.movement_threshold:
                if abs(dx) > abs(dy) * 1.5:
                    direction = "Right" if dx > 0 else "Left"
                elif abs(dy) > abs(dx) * 1.5:
                    direction = "Down" if dy > 0 else "Up"
                else:
                    direction = prev_dir if prev_dir else "Unknown"
                status = f"moving {direction}"
            else:
                status = "standing"
                direction = prev_dir

            self.objects[matched_id] = (bx, by, bw, bh, time.time(), cx, cy, direction)
            updated_ids.add(matched_id)

            cropped = frame[by:by+bh, bx:bx+bw].copy()
            class_name, prob = predict_single_image(self.model, cropped, self.device, self.class_names)

            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{matched_id} {class_name} {prob*100:.1f}%", (bx, by-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            tracked.append({
                "id": matched_id,
                "class": class_name,
                "prob": prob,
                "status": status,
                "box": (bx, by, bw, bh)
            })

        now = time.time()
        self.objects = {
            oid: data for oid, data in self.objects.items()
            if now - data[4] < 2.0
        }

        return frame, tracked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="0")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--weights", type=str, default="weights/best_model.pth")
    parser.add_argument("--train_dir", type=str, default="data/train")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = sorted([d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))])

    model = load_model(args.weights, num_classes=len(class_names), device=device)

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)

    pipeline = Pipeline(config, model, device, class_names)

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print("Error: cannot open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, tracked = pipeline.process_frame(frame)

        for obj in tracked:
            print(f"ID:{obj['id']} Class:{obj['class']}({obj['prob']*100:.1f}%) Status:{obj['status']} Box:{obj['box']}")

        if args.display:
            cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
