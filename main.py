from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import FluxKontextPipeline
import torch
from PIL import Image
import io

app = FastAPI()

# 1) Загружаем базовую модель
pipe = FluxKontextPipeline.from_pretrained(
    "kontext-community/flux-1-kontext-dev",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# 2) Загружаем LoRA-файл поверх
pipe.load_lora_weights(
    "kontext-community/relighting-kontext-dev-lora-v3",
    weight_name="relighting-kontext-dev-lora-v3.safetensors",
    adapter_name="lora"
)
pipe.set_adapters(["lora"], adapter_weights=[1.0])

@app.post("/relight")
async def relight(prompt: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = pipe(image=image, prompt=prompt).images[0]
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
