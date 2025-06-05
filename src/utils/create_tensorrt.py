import os
import sys
import argparse
import torch
from ultralytics import YOLO

def convert_to_tensorrt(model_path, task, half=False, workspace=8, dynamic=False):
    """
    Chuyển đổi mô hình YOLO sang TensorRT và lưu với thông tin task
    """
    print(f"Chuyển đổi {model_path} sang định dạng TensorRT{'(FP16)' if half else ''} cho task '{task}'")
    
    basename = os.path.splitext(model_path)[0]
    engine_path = f"{basename}_{task}.engine"
    
    if os.path.exists(engine_path):
        print(f"Engine đã tồn tại tại {engine_path}, xóa để tạo mới? (y/n)")
        response = input().lower()
        if response == 'y':
            os.remove(engine_path)
            print(f"Đã xóa engine cũ, tạo mới...")
        else:
            print(f"Giữ nguyên engine cũ")
            return engine_path
    
    # Load model 
    model = YOLO(model_path, task=task)
    
    try:
        model.export(format='engine', 
                    half=half, 
                    workspace=workspace, 
                    device=0 if torch.cuda.is_available() else "cpu",
                    dynamic=dynamic)
        
        default_export_path = model_path.replace('.pt', '.engine') 
        
        if os.path.exists(default_export_path) and default_export_path != engine_path:
            os.rename(default_export_path, engine_path)
            
        if os.path.exists(engine_path):
            print(f"Chuyển đổi thành công, engine lưu tại: {engine_path}")
            return engine_path
        else:
            print(f"Không tìm thấy engine sau khi chuyển đổi")
            return None
    except Exception as e:
        print(f"Lỗi khi chuyển đổi sang TensorRT: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo mô hình TensorRT từ mô hình YOLO")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến file .pt")
    parser.add_argument("--task", type=str, required=True, choices=["detect", "classify", "segment"],
                        help="Loại nhiệm vụ của mô hình")
    parser.add_argument("--half", action="store_true", help="Sử dụng FP16")
    parser.add_argument("--workspace", type=int, default=8, help="Workspace size (GB)")
    parser.add_argument("--dynamic", action="store_true", help="Sử dụng kích thước batch động")
    
    args = parser.parse_args()
    convert_to_tensorrt(args.model, args.task, args.half, args.workspace, args.dynamic)