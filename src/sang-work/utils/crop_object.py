import cv2
import os

output_folder = 'sample_object'
os.makedirs(output_folder, exist_ok=True)

start_point = None
end_point = None
cropping = False

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            end_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False

        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point

            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            cropped_image = image[y_min:y_max, x_min:x_max]

            filename = os.path.join(output_folder, f"sample_{x_min}_{y_min}_{x_max}_{y_max}.jpg")
            cv2.imwrite(filename, cropped_image)
            print(f"Lưu ảnh mẫu vào {filename})")
                  
# Đọc ảnh
image_path = r"D:\2_Hackathon\data_new\nighttime\cam_03_00660_jpg.rf.9fd7acc2a5afc406da18169ec9980c4b.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Không thể đọc ảnh. Kiểm tra lại đường dẫn ảnh.")
    exit()

# Tạo bản sao để hiển thị
clone = image.copy()

# Tạo cửa sổ và gán sự kiện chuột
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    # Hiển thị ảnh và vẽ hình chữ nhật khi kéo thả chuột
    display_image = clone.copy()
    if start_point and end_point and cropping:
        cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)

    cv2.imshow("Image", display_image)

    # Nhấn phím 'r' để reset ảnh
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        clone = image.copy()

    # Nhấn phím 'q' để thoát
    elif key == ord("q"):
        break

# Đóng tất cả cửa sổ
cv2.destroyAllWindows()