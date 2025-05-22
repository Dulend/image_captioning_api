from fastapi import FastAPI, UploadFile
from transformers import pipeline

app = FastAPI()
captioner = pipeline("image-to-text", model="bipin/tiny-vit-gpt2-image-captioning")

@app.post("/caption")
async def caption(file: UploadFile):
    result = captioner(await file.read())
    return {"caption": result[0]["generated_text"]}