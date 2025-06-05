import cv2
from ultralytics import solutions


cap = cv2.VideoCapture("/home/manhckv/vm7608/soict/__data/video2/0001_cut.mp4")
assert cap.isOpened(), "Error reading video file"

# Initialize object cropper object
cropper = solutions.ObjectCropper(
    # show=True,  # display the output
    model="/home/manhckv/vm7608/soict/weights/vehicle/best_public.pt",  # model for object cropping i.e yolo11x.pt.
    # classes=[
    #     0,
    #     2,
    # ],  # crop specific classes i.e. person and car with COCO pretrained model.
    conf=0.9,  # adjust confidence threshold for the objects.
    crop_dir="cropped-detections",  # set the directory name for cropped detections
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = cropper(im0)

    # print(results)  # access the output

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
