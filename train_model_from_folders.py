import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# -------------------------
# Helper: extract 6 raw features from dataframe
# -------------------------
def extract_6_features(df):
    df = df.select_dtypes(include=[np.number]).copy()
    if df.shape[1] == 0:
        return None

    # Remove timestamp-like strictly increasing column
    for col in df.columns:
        vals = df[col].values
        if vals.size > 1 and np.all(np.diff(vals) > 0):
            df = df.drop(columns=[col])
            break

    df = df.select_dtypes(include=[np.number]).copy()
    if df.shape[1] < 3:
        return None

    # First 3 = accel
    ax = df.iloc[:, 0].values
    ay = df.iloc[:, 1].values
    az = df.iloc[:, 2].values

    # Next 3 = magnetometer or zero-pad
    if df.shape[1] >= 6:
        mx = df.iloc[:, 3].values
        my = df.iloc[:, 4].values
        mz = df.iloc[:, 5].values
    else:
        n = len(ax)
        mx = np.zeros(n)
        my = np.zeros(n)
        mz = np.zeros(n)

    # Per-file mean features
    feat = np.array([
        np.mean(ax),
        np.mean(ay),
        np.mean(az),
        np.mean(mx),
        np.mean(my),
        np.mean(mz)
    ], dtype=float)

    return feat

# -------------------------
# Load all folder data
# -------------------------
data = []
labels = []
label_map = {"NORMAL": 0, "CHEWING": 1, "GRINDING": 2}

for folder, lab in label_map.items():
    files = sorted(glob.glob(f"{folder}/*.xlsx"))
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception as e:
            print("Skip (read error):", f, e)
            continue

        feat = extract_6_features(df)
        if feat is None:
            print("Skip (invalid cols):", f)
            continue

        data.append(feat)
        labels.append(lab)

X = np.array(data)
y = np.array(labels)

print("Samples loaded:", X.shape)

if X.shape[0] < 5:
    raise SystemExit("Not enough samples to train. Add more labeled files.")

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# CNN requires reshape: (samples, 6, 1)
# -------------------------
X_train_cnn = X_train.reshape(-1, 6, 1)
X_test_cnn  = X_test.reshape(-1, 6, 1)

y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat  = to_categorical(y_test, num_classes=3)

# -------------------------
# Build CNN model
# -------------------------
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', input_shape=(6,1)),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=2, activation='relu'),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------
# Train
# -------------------------
history = model.fit(
    X_train_cnn, y_train_cat,
    validation_data=(X_test_cnn, y_test_cat),
    epochs=60,
    batch_size=8,
    verbose=1
)

# -------------------------
# Evaluate
# -------------------------
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# Save model + scaler
# -------------------------
model.save("jaw_cnn_model.h5")
joblib.dump(scaler, "scaler_6feat.pkl")
print("\nSaved: jaw_cnn_model.h5 and scaler_6feat.pkl")
