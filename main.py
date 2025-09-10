from fastapi import FastAPI, UploadFile, File
from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("kontext-community/relighting-kontext-dev-lora-v3", token=HF_TOKEN)

app = FastAPI()

@app.post("/relight/")
async def relight_image(image: UploadFile = File(...)):
    image_bytes = await image.read()

    result = client.post(
        data={"inputs": image_bytes},
        params={"provider": "fal-ai"},
        headers={"Content-Type": "application/octet-stream"},
    )

    return result
