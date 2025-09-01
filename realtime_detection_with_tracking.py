import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    onnx_model_path = 'weights/model_epoch_1.onnx'
    net = cv2.dnn.readNetFromONNX(onnx_model_path)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    tracker = DeepSort(max_age=50)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    input_size = (416, 416)
    classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()
        outputs = outputs.reshape(1, 2 * (6 + 5), 7, 7)  # Sửa 7 nếu cần

        detections = []
        for a in range(2):
            for y in range(7):
                for x in range(7):
                    pred = outputs[0, a * (6 + 5):(a + 1) * (6 + 5), y, x]
                    conf = torch.sigmoid(torch.tensor(pred[0])).item()
                    if conf > 0.5:
                        box = pred[1:5]
                        cls_scores = torch.softmax(torch.tensor(pred[5:]), dim=0)
                        class_id = torch.argmax(cls_scores).item()
                        score = cls_scores[class_id].item() * conf

                        x_center = (x + box[0]) / 7 * frame.shape[1]
                        y_center = (y + box[1]) / 7 * frame.shape[0]
                        width = box[2] * frame.shape[1]
                        height = box[3] * frame.shape[0]
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        detections.append(([x1, y1, width, height], score, class_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = getattr(track, "det_class", 0)  # fallback nếu không có
            label = f"ID {track_id} {classes[class_id]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
