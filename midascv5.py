import cv2
import torch
from pygame import mixer
import numpy as np

# Download MiDaS
model_type = "DPT_Hybrid"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.eval()

# Initialize Pygame mixer
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')
alarm_playing = False

# Input transformational pipeline
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# Hook into OpenCV
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth_map = prediction.numpy()

    # Your logic to detect obstacles
    roi = depth_map[100:540]
    print(roi)
    less_than_threshold = np.any(roi < 2, axis=1)

    if np.any(less_than_threshold):
        if not alarm_playing:
            alarm_sound.play()
            alarm_playing = True
    else:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    output_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Display OpenCV frame
    cv2.imshow('Depth Map', output_norm)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
