import os

from PIL import Image
from pydantic import BaseModel, Field

from src_vlm.gemini import GeminiClient
from src_vlm.io_utils import read_text_file


class VehicleInformation(BaseModel):
    license_plate: str = Field(
        ...,
        description="The license plate number of the vehicle.",
    )
    description: str = Field(
        ...,
        description="A description of the vehicle.",
    )


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")

    system_prompt = read_text_file("src_vlm/prompt.md")
    # model_name = "gemma-3-27b-it"
    model_name = "gemini-2.0-flash-lite"

    client = GeminiClient(model_name=model_name, api_key=api_key)

    image = Image.open("crop_727.jpg")
    respone, usage = client.request(
        messages=[
            system_prompt,
            image,
        ],
        response_mime_type="application/json",
        response_schema=list[VehicleInformation],
    )

    print(respone)
