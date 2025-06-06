# Traffic Violation Detector with VLM License Plate Extraction

## 🎯 Simple VLM Integration

Chỉ thêm **3 dòng code** vào file gốc để có license plate extraction:

```python
# 1. Import addon
from vlm_addon import SimpleVLMProcessor

# 2. Initialize (in __init__)
self.vlm_processor = SimpleVLMProcessor(api_key="your_key", batch_size=5)

# 3. Add violation (in violation detection)
self.vlm_processor.add_violation(frame, bbox, track_id, frame_count, violation_info)
```

## 🚀 Cách chạy

### Option 1: Sử dụng script
```bash
./run_with_vlm.sh
```

### Option 2: Chạy trực tiếp
```bash
export GEMINI_API_KEY="your_api_key"

python traffic_violation_detector_with_vlm.py \
    --input video.mp4 \
    --output_video output.mp4 \
    --output_data violations.csv \
    --enable_vlm \
    --vlm_batch_size 5
```

## 📁 Files

- `traffic_violation_detector.py` - **File gốc** (không thay đổi gì)
- `traffic_violation_detector_with_vlm.py` - **File có VLM** (chỉ thêm 10 dòng)
- `vlm_addon.py` - **VLM processor** (background thread)
- `run_with_vlm.sh` - **Script chạy**

## 📊 Output

### Giống file gốc:
- `output.mp4` - Video với visualization  
- `violations.csv` - CSV detection data

### Thêm VLM:
- `violations_with_plates.json` - **Báo cáo với biển số**
- `violation_images/` - **Ảnh xe vi phạm**

## ⚡ Performance

- **✅ KHÔNG ảnh hưởng tốc độ detection chính**
- **✅ VLM chạy background thread**  
- **✅ Non-blocking batch processing**
- **✅ Batch size 5 = giảm 80% API calls**

## 🔧 Tùy chỉnh

```bash
--vlm_batch_size 3    # Real-time (nhanh)
--vlm_batch_size 10   # Cost-efficient (tiết kiệm)
```

## 📋 So sánh

| **Feature** | **Gốc** | **+ VLM** |
|---|---|---|
| Detection speed | ✅ | ✅ (same) |
| Memory usage | ✅ | ✅ (same) |  
| License plates | ❌ | ✅ |
| Violation images | ❌ | ✅ |
| API optimization | N/A | ✅ (batch) |

**Kết luận: Cùng performance, thêm license plate extraction! 🎉** 