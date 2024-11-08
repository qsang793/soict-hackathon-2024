import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# Load the model
model_path = "SOICT2024-VEHICLE-DETECTION-1CLS/run_0/weights/best.pt"
model = YOLO(model_path)

# Open the video file
video_path = "/home/manhckv/manhckv/soict/__temp/video.mp4"
cap = cv2.VideoCapture(video_path)

# Save video
output_dir = "__temp"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, os.path.basename(video_path) + "_tracked.mp4")

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

video_writer = cv2.VideoWriter(
    save_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)


# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model.track(
            frame,
            verbose=False,
            persist=True,
        )

        if results[0].boxes.id is None:
            # Write the frame to the output video
            video_writer.write(frame)
            continue

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot(font_size=8)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

        video_writer.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
