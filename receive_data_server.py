from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import uvicorn
import os

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "Weights"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload_weights")
async def upload_weights(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"status": "success", "filename": file.filename}

if __name__ == "__main__":
    url = ngrok.connect(5000)
    print(f"Public URL: {url}")
    uvicorn.run(app, host="0.0.0.0", port=5000)
