import numpy as np
from PIL import Image

from src_test.core import process_inpaint


if __name__ == "__main__":
    img_path = "/home/manhckv/manhckv/soict/src_test/test.jpg"

    input_image = Image.open(img_path)
    width, height = input_image.size

    bbox = [1000, 608, 1280, 720]
    x1, y1, x2, y2 = bbox

    im = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    im.paste((0, 0, 0, 0), (x1, y1, x2, y2))

    output = process_inpaint(np.array(input_image), np.array(im))
    img_output = Image.fromarray(output).convert("RGB")

    img_output.save("output.jpg")
