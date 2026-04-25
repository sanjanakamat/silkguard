import os

classes = set()

for file in os.listdir("data/processed/train/labels"):
    with open(f"data/processed/train/labels/{file}") as f:
        for line in f:
            classes.add(int(line.split()[0]))

print("Classes found:", classes)