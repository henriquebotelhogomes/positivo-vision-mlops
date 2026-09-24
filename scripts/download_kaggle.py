"""Script para download automatizado do dataset Kaggle akhatova/pcb-defects."""

import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def download_kaggle_dataset() -> Path:
    dest = Path("data/kaggle_raw")
    dest.mkdir(parents=True, exist_ok=True)

    # Se já existir a pasta descompactada PCB_DATASET, não precisa baixar de novo
    if (dest / "PCB_DATASET").exists() or (dest / "images").exists():
        print(f"[INFO] Dataset já presente em {dest.resolve()}")
        return dest

    print("[INFO] Autenticando com Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Baixando akhatova/pcb-defects (~1.9 GB) para {dest}...")
    api.dataset_download_files("akhatova/pcb-defects", path=str(dest), unzip=True)
    print(f"[SUCCESS] Download e descompactação concluídos em {dest.resolve()}")
    return dest


if __name__ == "__main__":
    download_kaggle_dataset()
