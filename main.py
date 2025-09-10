from fastapi import FastAPI, UploadFile, File
from huggingface_hub import InferenceClient
import os
from PIL import Image
import io
import base64

# Убедись, что переменная окружения HF_TOKEN задана в Render или Locally
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="kontext-community/relighting-kontext-dev-lora-v3",
    token=HF_TOKEN,
    provider="fal-ai",
)

app = FastAPI()

@app.post("/relight/")
async def relight_image(image: UploadFile = File(...)):
    image_bytes = await image.read()

    output_image = client.image_to_image(
        input_image=image_bytes,
        prompt="Relight this image in cinematic mood lighting"
    )

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    return {"result_base64": img_base64}
