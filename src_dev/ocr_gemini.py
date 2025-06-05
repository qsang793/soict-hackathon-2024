import os

from PIL import Image

from src_vlm.gemini import GeminiClient


api_key = os.getenv("GEMINI_API_KEY")

client = GeminiClient(
    model_name="gemini-2.0-flash-lite",
    api_key=api_key,
)


img_path = "crop_727.jpg"
img = Image.open(img_path)


system_prompt = "Trích xuất giá trị biển số xe từ ảnh biển số xe. Chỉ trả về giá trị biển số xe, không trả về bất kỳ thông tin nào khác."

response, usage = client.request(
    messages=[img],
    system_prompt=system_prompt,
)

print(response)
