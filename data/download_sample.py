"""
Data downloader and sample generator for ExtraSensory dataset.
Downloads minimal user features and generates continuous 25 Hz test sessions.
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

EXTRASENSORY_ZIP_URL = "http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip"
DATA_DIR = Path(__file__).resolve().parent
EXTRASENSORY_DIR = DATA_DIR / "extrasensory"

def download_minimal_extrasensory(max_users=2):
    """
    Downloads the ExtraSensory features/labels zip (214 MB) if needed,
    extracts only `max_users` user archives to conserve disk space,
    and removes the large zip file.
    """
    EXTRASENSORY_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(EXTRASENSORY_DIR.glob("*.features_labels.csv.gz"))
    if len(existing_files) >= max_users:
        print(f"[INFO] Already have {len(existing_files)} user files in {EXTRASENSORY_DIR}. Skipping download.")
        return [str(p) for p in existing_files[:max_users]]

    zip_path = DATA_DIR / "ExtraSensory_temp.zip"
    print(f"[DOWNLOAD] Downloading ExtraSensory per-user data from {EXTRASENSORY_ZIP_URL}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 200 == 0 or percent >= 100:
            sys.stdout.write(f"\r[PROGRESS] {percent}% ({count * block_size // (1024*1024)} MB / {total_size // (1024*1024)} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(EXTRASENSORY_ZIP_URL, zip_path, reporthook=reporthook)
    print("\n[DOWNLOAD] Download complete. Extracting target users...")

    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".features_labels.csv.gz") and not f.startswith("__MACOSX")]
        for member in csv_files[:max_users]:
            zip_ref.extract(member, EXTRASENSORY_DIR)
            extracted_path = EXTRASENSORY_DIR / member
            extracted_files.append(str(extracted_path))
            print(f"[EXTRACT] Extracted: {member}")

    # Remove temporary zip to keep storage low
    if zip_path.exists():
        zip_path.unlink()
        print("[CLEANUP] Deleted temporary zip file to conserve storage.")

    return extracted_files

if __name__ == "__main__":
    download_minimal_extrasensory(max_users=2)
