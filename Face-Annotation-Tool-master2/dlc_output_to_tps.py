import pandas as pd
import os

# Path setup
# Get path of the current script (where the .py file is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Relative path to the predicted folder
predicted_dir = os.path.join(script_dir, "labels", "predicted")

# CSV filename
csv_file = "image_predictions_DLC_Resnet50_fishApr29shuffle3_detector_200_snapshot_220.csv"
csv_path = os.path.join(predicted_dir, csv_file)

# Load CSV and ignore first 4 rows as header
df = pd.read_csv(csv_path, header=[0, 1, 2, 3])
data_rows = df

# Function to get fish ID from image path
def get_fish_id(path):
    return os.path.basename(path).split("_lat")[0]

# Process each row/image
for idx, row in data_rows.iterrows():
    image_path = row.iloc[0]  # Use iloc to avoid FutureWarning
    fish_id = get_fish_id(image_path)

    coords = []
    for i in range(1, 63, 3):  # Start from col 1, step through x/y pairs (21 labels × 3 columns)
        x = row.iloc[i]
        y = row.iloc[i + 1]
        coords.append((float(x), float(y)))

    # Write .tps content
    tps_lines = [f"LM={len(coords)}"]
    tps_lines += [f"{x:.4f} {y:.4f}" for x, y in coords]
    tps_lines += [
        f"IMAGE= Face-Annotation-Tool-master\\fish\\{fish_id}.jpg",
        f"ID= {fish_id}",
        "SCALE= -"
    ]

    tps_filename = f"{fish_id}_lat.tps"
    tps_path = os.path.join(predicted_dir, tps_filename)
    with open(tps_path, "w") as f:
        f.write("\n".join(tps_lines))

print("✓ All .tps files successfully generated.")
