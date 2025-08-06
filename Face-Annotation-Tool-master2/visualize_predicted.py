import os
import cv2

# Use current working directory as the actual project folder
project_dir = os.getcwd()
tps_dir = os.path.join(project_dir, "labels", "predicted")
image_dir = os.path.join(project_dir, "fish")
visualization_dir = os.path.join(tps_dir, "visualizations")

# Create output folder if it doesn't exist
os.makedirs(visualization_dir, exist_ok=True)

# Get all .tps files
tps_files = [f for f in os.listdir(tps_dir) if f.endswith(".tps")]

for tps_file in tps_files:
    tps_path = os.path.join(tps_dir, tps_file)

    with open(tps_path, "r") as f:
        lines = f.readlines()

    if not lines or not lines[0].startswith("LM="):
        print(f"Skipping invalid TPS file: {tps_file}")
        continue

    # Read coordinate lines
    num_landmarks = int(lines[0].split("=")[1])
    coords = [tuple(map(float, line.strip().split())) for line in lines[1:1 + num_landmarks]]

    # Image name matches tps file exactly, just with .jpg
    image_filename = tps_file.replace(".tps", ".jpg")
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue

    # Draw landmarks
    for x, y in coords:
        cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    # Save visualized output
    output_path = os.path.join(visualization_dir, image_filename.replace(".jpg", "_predicted.jpg"))
    cv2.imwrite(output_path, img)
    print(f"✅ Saved: {output_path}")

print("\n✓ All predictions visualized and saved.")
