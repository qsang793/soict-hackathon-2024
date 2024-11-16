import os

import cv2
from ultralytics import YOLO


# Load the model
model_path = "SOICT2024-VEHICLE-DETECTION-1CLS/run_1/weights/best.pt"
model = YOLO(model_path)

# Open the video file
video_path = "/home/manhckv/manhckv/soict/__video/src_2.mp4"
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

        # Visualize the results on the frame
        annotated_frame = results[0].plot(font_size=4, conf=False, line_width=1)
        video_writer.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
