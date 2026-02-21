from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import subprocess
import uuid

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===============================
# ARRANGEMENT CHECK ENDPOINT
# ===============================
@app.post("/arrangement")
async def check_arrangement(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        script_path = os.path.join(
            BASE_DIR,
            "SMARTSTOCK_AI2",
            "run_full_pipeline.py"
        )

        result = subprocess.run(
            ["python", script_path, "--image", image_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": result.stderr}
            )

        return {"message": "Arrangement check completed"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ===============================
# DEFECT CHECK ENDPOINT
# ===============================
@app.post("/defect")
async def check_defect(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        script_path = os.path.join(
            BASE_DIR,
            "SMARTSTOCKAI-AI",
            "infer.py"
        )

        result = subprocess.run(
            ["python", script_path, image_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": result.stderr}
            )

        return JSONResponse(content={"response": result.stdout})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )