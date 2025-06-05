import os

import cv2
from ultralytics import YOLO


# Load the YOLO11 model
model_path = "/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt"
model = YOLO(model_path, task="detect")

# Open the video file
video_path = "/home/manhckv/vm7608/soict/__data/video2/0006_cut.mp4"
cap = cv2.VideoCapture(video_path)

# Save video
output_dir = "__output"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, os.path.basename(video_path))

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

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            # verbose=False,
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
        annotated_frame = results[0].plot(
            font_size=5, conf=False, line_width=2, color_mode="instance"
        )
        video_writer.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
