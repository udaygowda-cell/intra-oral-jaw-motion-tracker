import os
import uuid
import shutil
import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# import your merged logic EXACTLY as-is
import merg   # your merged full script file

app = FastAPI()

UPLOAD_DIR = "uploads"
PLOT_DIR = "plots"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

app.mount("/plots", StaticFiles(directory=PLOT_DIR), name="plots")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ============================
# PAGE 1 — Upload + Analyze
# ============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:Arial;margin:40px;">
        <h2>Jaw Motion Analyzer</h2>

        <form action="/analyze" method="post" enctype="multipart/form-data">

            <input type="file" name="file" required>
            <br><br>

            <button type="submit" 
                style="padding:10px 20px;font-size:16px;">
                Analyze
            </button>
        </form>
    </body>
    </html>
    """


# ============================
# PAGE 2 — Run analysis + Show Plots + Prediction
# ============================
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile):

    file_id = str(uuid.uuid4())

    save_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- RUN YOUR EXACT CODE (plot + prediction) ----
    acc_plot_path, mag_plot_path, label, conf = merg.run_full_pipeline(save_path)

    # ---- HTML RETURN ----
    return f"""
    <html>
    <body style="font-family:Arial;margin:40px;">

        <h2>Analysis Result</h2>

        <h3>Prediction:</h3>
        <p><b>{label}</b></p>
        <p>Confidence: {conf:.4f}</p>

        <hr>

        <h3>Accelerometer 3D Plot</h3>
        <img src="/plots/{os.path.basename(acc_plot_path)}" width="600">

        <h3>Magnetometer 3D Plot</h3>
        <img src="/plots/{os.path.basename(mag_plot_path)}" width="600">

        <hr>

        <a href="/" style="font-size:18px;">Back</a>
    </body>
    </html>
    """
