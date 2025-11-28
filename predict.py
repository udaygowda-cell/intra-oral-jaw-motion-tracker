import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename


MODEL_PATH = "jaw_cnn_model.h5"
SCALER_PATH = "scaler_6feat.pkl"


def safe_load_df(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


# -------- strict column enforcement --------
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

    # magnetometer (optional)
    mag_map = {}
    for req, aliases in mag_aliases.items():
        for c, cu in cols_upper.items():
            if cu in aliases:
                mag_map[req] = c
                break

    return mapping, mag_map


# -------- extract same 6 features used during training --------
def extract_6_means(ax, ay, az, mx, my, mz):
    return np.array([
        np.mean(ax),
        np.mean(ay),
        np.mean(az),
        np.mean(mx),
        np.mean(my),
        np.mean(mz)
    ], dtype=float)


def predict_file(file_path, model, scaler):
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

    # CNN reshape: (1, 6, 1)
    feats_cnn = feats_scaled.reshape(1, 6, 1)

    probs = model.predict(feats_cnn)
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return pred, conf


# -------------------- MAIN --------------------
if __name__ == "__main__":
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select IMU Excel/CSV File",
        filetypes=[("Data files", "*.xlsx *.xls *.csv")]
    )

    if not file_path:
        print("No file selected.")
        raise SystemExit

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    pred, conf = predict_file(file_path, model, scaler)

    label_map = {0:"NORMAL", 1:"CHEWING", 2:"GRINDING"}

    print("\nSelected File:", file_path)
    print("FINAL PRED:", label_map[pred])
    print("Confidence:", conf)
