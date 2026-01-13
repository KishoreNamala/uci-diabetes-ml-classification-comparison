import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

zip_path = out_dir / "uci_diabetes_130us.zip"
csv_path = out_dir / "diabetic_data.csv"

# 1) Download ZIP
def download_zip():
    print("📥 Downloading UCI Diabetes dataset...")
    response = requests.get(DATASET_ZIP_URL, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"✅ Download complete, Saved ZIP to: {zip_path}")


# 2) Extract files
def extract_csv():
    print("📦 Extracting CSV...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    if not csv_path.exists():
        raise FileNotFoundError("❌ diabetic_data.csv not found after extraction")

    print("✅ Extraction complete")

# 3) Load + sanity check
def sanity_check():
    print("🔍 Running sanity checks...")
    df = pd.read_csv(csv_path)

    print("Shape:", df.shape)
    print("Columns:", len(df.columns))

    if "readmitted" not in df.columns:
        raise ValueError("❌ 'readmitted' column missing")

    print("\nReadmission distribution:")
    print(df["readmitted"].value_counts())

    print("\n✅ Dataset looks good")



# 4) Create binary target (<30 vs others)
# df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
# print("\nBinary target distribution:")
# print(df["readmitted_binary"].value_counts())

# Optional: save cleaned copy later (after preprocessing)
# df.to_csv(out_dir / "diabetic_data_with_target.csv", index=False)

#0) Start the download + extraction + sanity check process
def main():
    out_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        if not zip_path.exists():
            download_zip()
        extract_csv()
    else:
        print("ℹ️ Dataset already exists, skipping download")

    sanity_check()


if __name__ == "__main__":
    main()