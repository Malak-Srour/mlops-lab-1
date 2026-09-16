from pathlib import Path
from PIL import Image

CATEGORIES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
]

SPLITS = ["training", "evaluation", "validation"]
TARGET_SIZE = (128, 128)
MINI_LIMIT = 100

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")


def process_split(split: str):
    raw_split_dir = RAW_DIR / split
    counts = {cat: 0 for cat in CATEGORIES}

    for img_path in sorted(raw_split_dir.glob("*.jpg")):
        label_index = int(img_path.name.split("_")[0])
        category = CATEGORIES[label_index]

        img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)

        full_out_dir = PROCESSED_DIR / split / category
        full_out_dir.mkdir(parents=True, exist_ok=True)
        img.save(full_out_dir / img_path.name)

        if counts[category] < MINI_LIMIT:
            mini_out_dir = MINI_DIR / split / category
            mini_out_dir.mkdir(parents=True, exist_ok=True)
            img.save(mini_out_dir / img_path.name)
            counts[category] += 1


def main():
    for split in SPLITS:
        print(f"Processing {split}...")
        process_split(split)
    print("Done.")


if __name__ == "__main__":
    main()