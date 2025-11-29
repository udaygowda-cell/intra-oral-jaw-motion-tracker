import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import joblib
from tensorflow.keras.models import load_model
import uuid
import os


# ==========================================================
# PART 1: IMU FILE LOADING + CLEANING + PLOTTING
# ==========================================================

# This section is now handled by the Flask app
# We'll define the function to accept file_path as parameter

# The file processing code has been moved to the run_full_pipeline function


# ==========================================================
# PART 2: CNN PREDICTION MODULE (UNCHANGED)
# ==========================================================

MODEL_PATH = "jaw_cnn_model.h5"
SCALER_PATH = "scaler_6feat.pkl"

def safe_load_df(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def enforce_columns(df):
    accel_aliases = {
        "AX": ["AX", "ACCX", "A_X", "X", "ACC X", "ACC-X", "ACCX"],
        "AY": ["AY", "ACCY", "A_Y", "Y", "ACC Y", "ACC-Y", "ACCY"],
        "AZ": ["AZ", "ACCZ", "A_Z", "Z", "ACC Z", "ACC-Z", "ACCZ"]
    }

    mag_aliases = {
        "MX": ["MX", "MAGX", "M_X", "MAG X", "MAG-X"],
        "MY": ["MY", "MAGY", "M_Y", "MAG Y", "MAG-Y"],
        "MZ": ["MZ", "MAGZ", "M_Z", "MAG Z", "MAG-Y"]
    }

    cols_upper = {c: c.strip().upper() for c in df.columns}
    mapping = {}

    # accel
    for req, aliases in accel_aliases.items():
        found = None
        for c, cu in cols_upper.items():
            if cu in aliases:
                found = c
                break
        if not found:
            raise ValueError(f"Missing accelerometer column for {req}. Found: {df.columns}")
        mapping[req] = found

    # magnetometer optional
    mag_map = {}
    for req, aliases in mag_aliases.items():
        for c, cu in cols_upper.items():
            if cu in aliases:
                mag_map[req] = c
                break

    return mapping, mag_map


def extract_6_means(ax, ay, az, mx, my, mz):
    return np.array([
        np.mean(ax),
        np.mean(ay),
        np.mean(az),
        np.mean(mx),
        np.mean(my),
        np.mean(mz)
    ], dtype=float)


def predict_file_cnn(file_path, model, scaler):
    df = safe_load_df(file_path)

    mapping, mag_map = enforce_columns(df)

    ax = df[mapping["AX"]].values.astype(float)
    ay = df[mapping["AY"]].values.astype(float)
    az = df[mapping["AZ"]].values.astype(float)
    n = len(ax)

    mx = df[mag_map.get("MX", None)].values.astype(float) if "MX" in mag_map else np.zeros(n)
    my = df[mag_map.get("MY", None)].values.astype(float) if "MY" in mag_map else np.zeros(n)
    mz = df[mag_map.get("MZ", None)].values.astype(float) if "MZ" in mag_map else np.zeros(n)

    feats = extract_6_means(ax, ay, az, mx, my, mz)
    feats_scaled = scaler.transform([feats])

    feats_cnn = feats_scaled.reshape(1, 6, 1)

    probs = model.predict(feats_cnn)
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return pred, conf


# LOAD MODEL + SCALER
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# The prediction code has been moved to the run_full_pipeline function

import matplotlib
matplotlib.use("Agg")

def run_full_pipeline(file_path):
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Load and process the file
    if file_path.endswith(".csv"):
        raw = pd.read_csv(file_path, header=None, dtype=str)
    else:
        raw = pd.read_excel(file_path, header=None, dtype=str)

    # DETECT HEADER ROW
    def is_header_row(row):
        for v in row:
            v = str(v).strip()
            if any(c.isalpha() for c in v):
                return True
        return False

    if is_header_row(raw.iloc[0]):
        df = raw.iloc[1:].reset_index(drop=True)
    else:
        df = raw.copy()

    # FORCE 8 COLUMNS
    if df.shape[1] < 8:
        raise ValueError("File has fewer than 8 columns. Expected 8.")

    df = df.iloc[:, :8]
    df.columns = ["timestamp", "millis", "ax", "ay", "az", "mx", "my", "mz"]

    # CLEAN NUMERIC
    def clean_numeric(col):
        col = col.astype(str)
        col = col.str.replace(r"[^\d.+-Ee]", "", regex=True)
        return pd.to_numeric(col, errors="coerce")

    for c in ["ax", "ay", "az", "mx", "my", "mz"]:
        df[c] = clean_numeric(df[c])

    imu_cols = ["ax", "ay", "az", "mx", "my", "mz"]
    df = df.dropna(subset=imu_cols, how="all").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No numeric data found in file.")

    # SMOOTHING
    def smooth(x):
        if len(x) < 7:
            return x.to_numpy()
        return savgol_filter(x, window_length=7, polyorder=2)

    for c in imu_cols:
        df[c + "_s"] = smooth(df[c])

    # REPEATED POINT DETECTION
    def detect_repeats(x, y, z):
        repeats = []
        for i in range(1, len(x)):
            if (
                x.iloc[i] == x.iloc[i - 1]
                and y.iloc[i] == y.iloc[i - 1]
                and z.iloc[i] == z.iloc[i - 1]
            ):
                repeats.append(i)
        return repeats

    acc_repeats = detect_repeats(df["ax_s"], df["ay_s"], df["az_s"])
    mag_repeats = detect_repeats(df["mx_s"], df["my_s"], df["mz_s"])

    # run your full IMU clean + plot
    acc_plot = "plots/acc_" + str(uuid.uuid4()) + ".png"
    mag_plot = "plots/mag_" + str(uuid.uuid4()) + ".png"

    # Create plots
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    plot_3d_to_ax(ax1, df["ax_s"], df["ay_s"], df["az_s"], acc_repeats, "Accelerometer 3D")
    plt.savefig(acc_plot)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    plot_3d_to_ax(ax2, df["mx_s"], df["my_s"], df["mz_s"], mag_repeats, "Magnetometer 3D")
    plt.savefig(mag_plot)
    plt.close(fig2)

    # run your CNN code EXACTLY as-is
    pred, conf = predict_file_cnn(file_path, model, scaler)

    label_map = {0:"NORMAL", 1:"CHEWING", 2:"GRINDING"}
    label = label_map[pred]

    return acc_plot, mag_plot, label, conf

# Modified plotting function that takes an axis parameter
def plot_3d_to_ax(ax, x, y, z, repeats, title):
    ax.plot(x, y, z, linewidth=1.5)

    if repeats:
        ax.scatter(
            x.iloc[repeats],
            y.iloc[repeats],
            z.iloc[repeats],
            color="red",
            s=20,
            label="Repeated points",
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if repeats:
        ax.legend()
    plt.tight_layout()
