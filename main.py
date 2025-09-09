from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from diffusers import FluxKontextPipeline
import torch, io
from PIL import Image

app = FastAPI()

pipe = FluxKontextPipeline.from_pretrained(
    "kontext-community/relighting-kontext-dev-lora-v3",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/relight")
async def relight(prompt: str = Form(...), file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = pipe(image=img, prompt=prompt).images[0]
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
