"""Ingestão e Extração de Patches do Dataset Kaggle (PKU PCB Defects).

Lê as anotações Pascal VOC (.xml) de cada defeito, calcula o centro geométrico (cx, cy)
e aplica Window Clamping para extrair patches de 224x224 garantindo que nenhum defeito
próximo às bordas seja cortado:
  x1 = max(0, min(w - patch_size, cx - half))
  y1 = max(0, min(h - patch_size, cy - half))
"""

import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Mapeamento oficial dos 6 defeitos do Kaggle para as classes canônicas do projeto
KAGGLE_FOLDER_TO_CLASS = {
    "Missing_hole": "defect_missing_hole",
    "Mouse_bite": "defect_mousebite",
    "Open_circuit": "defect_open",
    "Short": "defect_short",
    "Spur": "defect_spur",
    "Spurious_copper": "defect_spurious_copper",
}

KAGGLE_NAME_TO_CLASS = {
    "missing_hole": "defect_missing_hole",
    "mouse_bite": "defect_mousebite",
    "mousebite": "defect_mousebite",
    "open_circuit": "defect_open",
    "open": "defect_open",
    "short": "defect_short",
    "spur": "defect_spur",
    "spurious_copper": "defect_spurious_copper",
}


def parse_xml_annotation(xml_path: Path) -> list[dict]:
    """Extrai todas as bounding boxes de defeitos de um arquivo XML Pascal VOC."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_elem is None or bndbox is None:
            continue

        raw_name = name_elem.text.strip().lower() if name_elem.text else ""
        cls_name = KAGGLE_NAME_TO_CLASS.get(raw_name)
        if not cls_name:
            continue

        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append({
                "class": cls_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "cx": (xmin + xmax) // 2,
                "cy": (ymin + ymax) // 2,
            })
        except (ValueError, TypeError, AttributeError):
            continue

    return boxes


def extract_clamped_patch(
    img: Image.Image,
    cx: int,
    cy: int,
    patch_size: int = 224,
) -> Image.Image:
    """Extrai patch de 224x224 com clamping rigoroso nas 4 bordas da imagem."""
    w, h = img.size
    half = patch_size // 2

    # Clamping da janela deslizante dentro dos limites físicos da imagem
    x1 = max(0, min(w - patch_size, cx - half))
    y1 = max(0, min(h - patch_size, cy - half))
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    return img.crop((x1, y1, x2, y2)).convert("RGB")


def is_patch_defect_free(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    defect_boxes: list[dict],
    margin: int = 20,
) -> bool:
    """Verifica se um recorte retangular está completamente livre de defeitos."""
    for b in defect_boxes:
        # Verifica se há sobreposição (interseção) com margem de segurança
        bx1 = b["xmin"] - margin
        by1 = b["ymin"] - margin
        bx2 = b["xmax"] + margin
        by2 = b["ymax"] + margin

        if not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2):
            return False  # Sobrepôs com um defeito
    return True


def process_kaggle_dataset(
    kaggle_dir: Path,
    output_dir: Path,
    samples_per_class: int = 300,
    patch_size: int = 224,
    seed: int = 42,
) -> dict[str, int]:
    """Processa o dataset Kaggle e extrai patches anotados para cada classe."""
    random.seed(seed)

    # Procura a pasta PCB_DATASET ou images/Annotations
    ann_dir = None
    img_dir = None
    for candidate in [kaggle_dir / "PCB_DATASET", kaggle_dir]:
        if (candidate / "Annotations").exists():
            ann_dir = candidate / "Annotations"
            img_dir = candidate / "images"
            break

    if not ann_dir or not ann_dir.exists():
        raise FileNotFoundError(f"Diretório Annotations não encontrado em {kaggle_dir}")

    print(f"[INFO] Inspecionando anotações em: {ann_dir}")
    print(f"[INFO] Inspecionando imagens em: {img_dir}")

    # Localiza pares (imagem, anotações)
    records = []
    for xml_file in ann_dir.rglob("*.xml"):
        base_name = xml_file.stem
        # Procura a imagem correspondente em img_dir
        matching_imgs = list(img_dir.rglob(f"{base_name}.jpg")) + list(img_dir.rglob(f"{base_name}.png"))
        if matching_imgs:
            boxes = parse_xml_annotation(xml_file)
            if boxes:
                records.append({
                    "img_path": matching_imgs[0],
                    "xml_path": xml_file,
                    "boxes": boxes,
                })

    print(f"[SUCCESS] {len(records)} placas com defeitos anotados encontradas no Kaggle.")

    # Embaralha para split limpo
    random.shuffle(records)

    class_counts = {
        "defect_missing_hole": 0,
        "defect_mousebite": 0,
        "defect_open": 0,
        "defect_short": 0,
        "defect_spur": 0,
        "defect_spurious_copper": 0,
        "normal": 0,
    }

    # Prepara diretórios de saída
    for cls_name in class_counts:
        (output_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # 1. Extração de Defeitos com Clamping
    for rec in records:
        try:
            img = Image.open(rec["img_path"])
        except Exception:
            continue

        for box in rec["boxes"]:
            cls_name = box["class"]
            if class_counts[cls_name] < samples_per_class:
                patch = extract_clamped_patch(img, box["cx"], box["cy"], patch_size=patch_size)
                save_path = output_dir / cls_name / f"kaggle_{cls_name}_{class_counts[cls_name]:04d}.png"
                patch.save(save_path)
                class_counts[cls_name] += 1

        # 2. Extração de Áreas Normais da mesma placa (onde não há caixas de defeito)
        if class_counts["normal"] < samples_per_class:
            w, h = img.size
            # Tenta posições em grade
            for gx in range(0, w - patch_size, 180):
                for gy in range(0, h - patch_size, 180):
                    if class_counts["normal"] >= samples_per_class:
                        break
                    if is_patch_defect_free(gx, gy, gx + patch_size, gy + patch_size, rec["boxes"]):
                        patch = img.crop((gx, gy, gx + patch_size, gy + patch_size)).convert("RGB")
                        save_path = output_dir / "normal" / f"kaggle_normal_{class_counts['normal']:04d}.png"
                        patch.save(save_path)
                        class_counts["normal"] += 1

    print(f"[SUCCESS] Patches extraídos do Kaggle: {class_counts}")
    return class_counts


if __name__ == "__main__":
    raw_kaggle = Path("data/kaggle_raw")
    out_patches = Path("data/kaggle_patches")
    process_kaggle_dataset(raw_kaggle, out_patches, samples_per_class=250)
