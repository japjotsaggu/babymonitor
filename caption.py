import torch
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import requests
import numpy as np

def detect_and_caption(video_path):
    # YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # BLIP model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    blip_model = blip_decoder(pretrained=model_url, image_size=384, vit='base').to(device)
    blip_model.eval()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #object detection w YOLO
        results = model(frame)
        results.render()

        # Process detections
        for det in results.pred[0]:
            label = f'{det[5]} {det[4]:.2f}'
            cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(det[0]), int(det[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # image captioning using BLIP
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            caption = blip_model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)[0]
            print('caption:', caption)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'videoplayback.mp4'
    detect_and_caption(video_path)
