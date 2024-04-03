import torch
import cv2
from torchvision.transforms.functional import InterpolationMode
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox

def detect_objects(video_path):
    weights = 'yolov5x.pt'  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights, map_location=device)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img).float().to(device) / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)

        # Process detections
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'babymonitor/videoplayback.mp4'
    detect_objects(video_path)
