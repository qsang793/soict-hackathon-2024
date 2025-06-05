import cv2
from ultralytics import solutions


cap = cv2.VideoCapture("/home/manhckv/vm7608/soict/__data/video2/0001_cut.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    "speed_management.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)


# Initialize speed estimation object
speedestimator = solutions.SpeedEstimator(
    model="/home/manhckv/vm7608/soict/weights/vehicle/best_private.pt",  # path to the YOLO11 model file.
    fps=fps,
    max_speed=120,
    max_hist=3,
    conf=0.5,
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = speedestimator(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
