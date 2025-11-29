import os
import uuid
import shutil
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import merg

app = Flask(__name__)

UPLOAD_DIR = "uploads"
PLOT_DIR = "plots"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    
    if file:
        file_id = str(uuid.uuid4())
        save_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"
        file.save(save_path)
        
        # Run analysis
        acc_plot_path, mag_plot_path, label, conf = merg.run_full_pipeline(save_path)
        
        # Extract filenames for template
        acc_plot_filename = os.path.basename(acc_plot_path)
        mag_plot_filename = os.path.basename(mag_plot_path)
        
        return render_template("result.html", 
                             label=label, 
                             confidence=conf,
                             acc_plot=acc_plot_filename,
                             mag_plot=mag_plot_filename)

@app.route("/plots/<filename>")
def plot_file(filename):
    return send_from_directory(PLOT_DIR, filename)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)