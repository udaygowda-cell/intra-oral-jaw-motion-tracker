import pandas as pd
import numpy as np
import glob
import os
import shutil

# -------------------------------------
# Create output folders
# -------------------------------------
os.makedirs("CLENCHING", exist_ok=True)
os.makedirs("CHEWING", exist_ok=True)
os.makedirs("GRINDING", exist_ok=True)

# -------------------------------------
# Function to classify a file
# -------------------------------------
def classify_file(file_name):

    df = pd.read_excel(file_name)

    # Take only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Calculate standard deviation (movement)
    std_mean = df.std().mean()

    # Classification logic
    if std_mean < 0.2:
        label = "CLENCHING"

    elif 0.2 <= std_mean <= 1.0:
        label = "CHEWING"

    else:
        label = "GRINDING"

    return label, std_mean


# -------------------------------------
# LOAD ALL FILES
# -------------------------------------
files = sorted(
    glob.glob("KKub*.xlsx") +
    glob.glob("Kub*.xlsx") +
    glob.glob("SF_R*.xlsx") +
    glob.glob("SF22_R*.xlsx") +
    glob.glob("v3Dyn*.xlsx") +
    glob.glob("P2_v3_Kre*.xlsx") +
    glob.glob("B_values.xlsx")
)

if not files:
    print("❌ No files found in the folder!")
    exit()

print("\n========== MASTER FILE CLASSIFICATION ==========\n")

results = []

# -------------------------------------
# Classify each file
# -------------------------------------
for file in files:

    label, strength = classify_file(file)

    results.append([file, label, strength])

    print(f"{file:25} ---> {label:10} (STD = {strength:.4f})")

    # Move to folder
    if file != "B_values.xlsx":   # optional: don't move this if you don't want
        shutil.move(file, os.path.join(label, file))


# -------------------------------------
# Save result to Excel
# -------------------------------------
result_df = pd.DataFrame(results, columns=[
    "File Name",
    "Predicted Activity",
    "STD_Mean"
])

result_df.to_excel("MASTER_Classification_Result.xlsx", index=False)

print("\n✅ DONE!")
print("✅ All files classified & sorted into folders")
print("✅ Excel saved as: MASTER_Classification_Result.xlsx")