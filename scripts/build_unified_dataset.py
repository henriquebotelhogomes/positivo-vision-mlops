"""Script de Construção do Dataset Unificado Multi-Fonte (DeepPCB + Kaggle).

Combina patches de alta fidelidade:
1. DeepPCB (Peking University, escâner AOI cinza)
2. PKU PCB (Kaggle akhatova/pcb-defects, fotografia macro RGB verde)

Estrutura 7 classes nominais completas:
  0. normal (CONFORME)
  1. defect_short (CURTO-CIRCUITO)
  2. defect_open (TRILHA ROMPIDA)
  3. defect_missing_hole (FURO DE VIA AUSENTE)
  4. defect_mousebite (MORDEDURA DE TRILHA)
  5. defect_spur (ESPORÃO / REBARBA)
  6. defect_spurious_copper (COBRE ESPÚRIO / ILHA ISOLADA)
"""

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLASSES = [
    "normal",
    "defect_short",
    "defect_open",
    "defect_missing_hole",
    "defect_mousebite",
    "defect_spur",
    "defect_spurious_copper",
]

# Kaggle XML names
KAGGLE_MAP = {
    "missing_hole": "defect_missing_hole",
    "mouse_bite": "defect_mousebite",
    "mousebite": "defect_mousebite",
    "open_circuit": "defect_open",
    "open": "defect_open",
    "short": "defect_short",
    "spur": "defect_spur",
    "spurious_copper": "defect_spurious_copper",
}

# DeepPCB type IDs
DEEPPCB_MAP = {
    1: "defect_open",
    2: "defect_short",
    3: "defect_mousebite",
    4: "defect_spur",
    5: "defect_spurious_copper",
    6: "defect_missing_hole",
}


def extract_clamped_patch(img: Image.Image, cx: int, cy: int, patch_size: int = 224) -> Image.Image:
    """Extrai patch de 224x224 com clamping rigoroso nas bordas."""
    w, h = img.size
    half = patch_size // 2
    x1 = max(0, min(w - patch_size, cx - half))
    y1 = max(0, min(h - patch_size, cy - half))
    return img.crop((x1, y1, x1 + patch_size, y1 + patch_size)).convert("RGB")


def parse_kaggle_annotations(ann_dir: Path) -> list[dict]:
    """Parseia todas as anotações do Kaggle."""
    records = []
    for xml_file in ann_dir.rglob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_elem is not None and bndbox is not None:
                raw_name = name_elem.text.strip().lower() if name_elem.text else ""
                cls = KAGGLE_MAP.get(raw_name)
                if cls:
                    try:
                        xmin = int(float(bndbox.find("xmin").text))
                        ymin = int(float(bndbox.find("ymin").text))
                        xmax = int(float(bndbox.find("xmax").text))
                        ymax = int(float(bndbox.find("ymax").text))
                        boxes.append({
                            "class": cls,
                            "cx": (xmin + xmax) // 2,
                            "cy": (ymin + ymax) // 2,
                            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        })
                    except Exception:
                        pass
        if boxes:
            records.append({"xml": xml_file, "boxes": boxes})
    return records


def parse_deeppcb_annotations(data_dir: Path) -> list[dict]:
    """Parseia anotações do DeepPCB."""
    records = []
    groups = sorted([g for g in data_dir.iterdir() if g.is_dir() and g.name.startswith("group")])
    for g in groups:
        g_num = g.name.replace("group", "")
        img_dir = g / g_num
        not_dir = g / f"{g_num}_not"
        if not img_dir.exists() or not not_dir.exists():
            continue
        for test_path in sorted(img_dir.glob("*_test.jpg")):
            base_id = test_path.stem.replace("_test", "")
            temp_path = img_dir / f"{base_id}_temp.jpg"
            ann_path = not_dir / f"{base_id}.txt"
            if temp_path.exists() and ann_path.exists():
                boxes = []
                with open(ann_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            x1, y1, x2, y2, dtype = map(int, parts[:5])
                            cls = DEEPPCB_MAP.get(dtype)
                            if cls:
                                boxes.append({
                                    "class": cls,
                                    "cx": (x1 + x2) // 2,
                                    "cy": (y1 + y2) // 2,
                                })
                if boxes:
                    records.append({
                        "test_path": test_path,
                        "temp_path": temp_path,
                        "boxes": boxes,
                    })
    return records


def build_unified_dataset(
    output_root: Path,
    kaggle_per_class: int = 250,
    deeppcb_per_class: int = 250,
    val_ratio: float = 0.20,
    pool_per_class: int = 12,
    seed: int = 42,
) -> None:
    random.seed(seed)
    raw_dir = output_root / "data" / "raw"
    pool_dir = output_root / "data" / "sample_pool"

    # Limpa diretórios antigos para garantir dados 100% frescos
    for d in [raw_dir / "train", raw_dir / "val", pool_dir]:
        if d.exists():
            shutil.rmtree(d)

    for split in ["train", "val"]:
        for c in CLASSES:
            (raw_dir / split / c).mkdir(parents=True, exist_ok=True)
    for c in CLASSES + ["unknown"]:
        (pool_dir / c).mkdir(parents=True, exist_ok=True)

    print("[1/4] Extraindo patches do Kaggle (Verde RGB)...")
    kaggle_patches = {c: [] for c in CLASSES}
    kaggle_base = output_root / "data" / "kaggle_raw" / "PCB_DATASET"
    ann_records = parse_kaggle_annotations(kaggle_base / "Annotations")
    random.shuffle(ann_records)

    # 1. Defeitos Kaggle
    for rec in ann_records:
        stem = rec["xml"].stem
        img_candidates = list((kaggle_base / "images").rglob(f"{stem}.jpg")) + list((kaggle_base / "images").rglob(f"{stem}.png"))
        if not img_candidates:
            continue
        try:
            img = Image.open(img_candidates[0])
        except Exception:
            continue

        for b in rec["boxes"]:
            cls = b["class"]
            if len(kaggle_patches[cls]) < kaggle_per_class:
                patch = extract_clamped_patch(img, b["cx"], b["cy"])
                kaggle_patches[cls].append(patch)

    # 2. Normais Kaggle (Extraídos dos templates PCB_USED)
    used_dir = kaggle_base / "PCB_USED"
    if used_dir.exists():
        for golden_img_path in sorted(used_dir.glob("*.JPG")):
            if len(kaggle_patches["normal"]) >= kaggle_per_class:
                break
            try:
                g_img = Image.open(golden_img_path)
                w, h = g_img.size
                for gx in range(100, w - 224, 250):
                    for gy in range(100, h - 224, 250):
                        if len(kaggle_patches["normal"]) >= kaggle_per_class:
                            break
                        patch = g_img.crop((gx, gy, gx + 224, gy + 224)).convert("RGB")
                        arr = np.array(patch.convert("L"))
                        if arr.mean() > 240 or arr.std() < 10:
                            continue
                        kaggle_patches["normal"].append(patch)
            except Exception:
                continue

    print(f"  -> Kaggle concluído: { {k: len(v) for k, v in kaggle_patches.items()} }")

    print("[2/4] Extraindo patches do DeepPCB (Cinza AOI)...")
    deeppcb_patches = {c: [] for c in CLASSES}
    deeppcb_records = parse_deeppcb_annotations(output_root / "data" / "deeppcb_raw" / "PCBData")
    random.shuffle(deeppcb_records)

    for rec in deeppcb_records:
        test_img = None
        for b in rec["boxes"]:
            cls = b["class"]
            if len(deeppcb_patches[cls]) < deeppcb_per_class:
                if test_img is None:
                    test_img = Image.open(rec["test_path"])
                patch = extract_clamped_patch(test_img, b["cx"], b["cy"])
                deeppcb_patches[cls].append(patch)

        # Normais DeepPCB (de _temp.jpg)
        if len(deeppcb_patches["normal"]) < deeppcb_per_class:
            temp_img = Image.open(rec["temp_path"])
            grid_positions = [(112, 112), (320, 112), (112, 320), (320, 320), (224, 224)]
            for gx, gy in grid_positions:
                if len(deeppcb_patches["normal"]) >= deeppcb_per_class:
                    break
                patch = extract_clamped_patch(temp_img, gx, gy)
                deeppcb_patches["normal"].append(patch)

    print(f"  -> DeepPCB concluído: { {k: len(v) for k, v in deeppcb_patches.items()} }")

    print("[3/4] Unificando, estratificando e salvando em Train/Val (80/20)...")
    summary = {}
    for cls in CLASSES:
        all_patches = kaggle_patches[cls] + deeppcb_patches[cls]
        random.shuffle(all_patches)

        # Reserva para o Sample Pool da Demo (para a entrevista)
        pool_samples = all_patches[:pool_per_class]
        remaining = all_patches[pool_per_class:]

        for idx, p in enumerate(pool_samples, 1):
            p.save(pool_dir / cls / f"sample_{cls}_{idx:02d}.png")

        # Split 80/20 Train/Val
        val_count = int(len(remaining) * val_ratio)
        val_set = remaining[:val_count]
        train_set = remaining[val_count:]

        for idx, p in enumerate(train_set, 1):
            p.save(raw_dir / "train" / cls / f"{cls}_train_{idx:04d}.png")
        for idx, p in enumerate(val_set, 1):
            p.save(raw_dir / "val" / cls / f"{cls}_val_{idx:04d}.png")

        summary[cls] = {"train": len(train_set), "val": len(val_set), "pool": len(pool_samples)}

    # Popula unknown no sample pool (anomalias sintéticas/deformadas para OOD test)
    for idx in range(1, pool_per_class + 1):
        if idx <= len(kaggle_patches["defect_mousebite"]):
            # Usa patch de mousebite com inversão de cor ou rotação estranha
            p = kaggle_patches["defect_mousebite"][idx - 1].transpose(Image.Transpose.ROTATE_90)
            p.save(pool_dir / "unknown" / f"sample_unknown_{idx:02d}.png")

    print("[4/4] Dataset Unificado gerado com sucesso!")
    print("=" * 60)
    for cls, cnt in summary.items():
        print(f"  {cls:<25}: Train={cnt['train']} | Val={cnt['val']} | Pool={cnt['pool']}")
    print("=" * 60)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    build_unified_dataset(project_root)
