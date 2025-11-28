import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# FILE PICKER
# ==========================================================
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select IMU CSV File",
    filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
)

if not file_path:
    print("No file selected.")
    exit()

# ==========================================================
# LOAD RAW FILE (HEADER UNKNOWN)
# ==========================================================
if file_path.endswith(".csv"):
    raw = pd.read_csv(file_path, header=None, dtype=str)
else:
    raw = pd.read_excel(file_path, header=None, dtype=str)

print("RAW SHAPE:", raw.shape)
print(raw.head())

# ==========================================================
# DETECT HEADER ROW
# If first row contains any non-numeric → treat it as header and skip.
# ==========================================================
def is_header_row(row):
    for v in row:
        v = str(v).strip()
        # if value contains letters → header
        if any(c.isalpha() for c in v):
            return True
    return False

if is_header_row(raw.iloc[0]):
    df = raw.iloc[1:].reset_index(drop=True)
else:
    df = raw.copy()

# ==========================================================
# FORCE 8 COLUMNS
# ==========================================================
if df.shape[1] < 8:
    raise ValueError("File has fewer than 8 columns. Expected 8.")

df = df.iloc[:, :8]   # use only first 8 columns
df.columns = ["timestamp", "millis", "ax", "ay", "az", "mx", "my", "mz"]

# ==========================================================
# CLEAN AND CONVERT TO NUMERIC
# Removes units, spaces, non-numeric garbage.
# ==========================================================
def clean_numeric(col):
    col = col.astype(str)
    col = col.str.replace(r"[^\d.+-Ee]", "", regex=True)  # keep numbers only
    return pd.to_numeric(col, errors="coerce")

for c in ["ax", "ay", "az", "mx", "my", "mz"]:
    df[c] = clean_numeric(df[c])

# drop rows where all IMU columns are missing
imu_cols = ["ax", "ay", "az", "mx", "my", "mz"]
df = df.dropna(subset=imu_cols, how="all").reset_index(drop=True)

print("CLEANED SHAPE:", df.shape)
print(df.head())

if len(df) == 0:
    print("No numeric data found in file.")
    exit()

# ==========================================================
# SMOOTHING
# ==========================================================
def smooth(x):
    if len(x) < 7:
        return x.to_numpy()
    return savgol_filter(x, window_length=7, polyorder=2)

for c in imu_cols:
    df[c + "_s"] = smooth(df[c])

# ==========================================================
# REPEATED POINT DETECTION
# ==========================================================
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

print("ACC REPEATS:", len(acc_repeats))
print("MAG REPEATS:", len(mag_repeats))

# ==========================================================
# 3D PLOT FUNCTION
# ==========================================================
def plot_3d(x, y, z, repeats, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

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
    plt.show()

# ==========================================================
# PLOT
# ==========================================================
plot_3d(df["ax_s"], df["ay_s"], df["az_s"], acc_repeats, "Accelerometer 3D")
plot_3d(df["mx_s"], df["my_s"], df["mz_s"], mag_repeats, "Magnetometer 3D")
