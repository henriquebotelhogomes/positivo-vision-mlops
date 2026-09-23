"""Ingestão e Curadoria Industrial do Dataset DeepPCB (Peking University).

Extrai patches de 224x224 a partir de fotografias reais de PCBs e gabaritos:
1. NORMAL: Recortes de imagens gabarito (_temp.jpg) 100% livres de defeitos
2. DEFECT_SHORT: Recortes de _test.jpg centrados no defeito tipo 2 (curto)
3. DEFECT_OPEN: Recortes de _test.jpg centrados no defeito tipo 1 (trilha aberta)
4. DEFECT_MISSING_HOLE: Recortes de _test.jpg centrados no defeito tipo 6 (pin-hole / furo ausente)
5. DEFECT_SPURIOUS: Recortes de _test.jpg centrados nos defeitos tipos 4 e 5 (spur / copper)
6. UNKNOWN_ANOMALY: Recortes de _test.jpg com tipo 3 (mousebite / mordedura) para Open-Set test
"""

import random
import shutil
import sys
from pathlib import Path

from PIL import Image

# Força UTF-8 no stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_annotations_and_images(pcb_data_dir: Path) -> dict:
    """Mapeia todos os pares de imagens (test, temp) e suas anotações no DeepPCB."""
    records = []
    groups = [g for g in pcb_data_dir.iterdir() if g.is_dir() and g.name.startswith("group")]

    for g in sorted(groups):
        # Cada grupo tem subpasta com o nome do grupo e subpasta _not
        g_name = g.name.replace("group", "")
        img_dir = g / g_name
        not_dir = g / f"{g_name}_not"

        if not img_dir.exists() or not not_dir.exists():
            continue

        test_imgs = sorted(list(img_dir.glob("*_test.jpg")))
        for test_path in test_imgs:
            base_id = test_path.stem.replace("_test", "")
            temp_path = img_dir / f"{base_id}_temp.jpg"
            ann_path = not_dir / f"{base_id}.txt"

            if temp_path.exists() and ann_path.exists():
                defects = []
                with open(ann_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            x1, y1, x2, y2, dtype = map(int, parts[:5])
                            defects.append({
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "type": dtype,
                            })
                records.append({
                    "id": base_id,
                    "group": g.name,
                    "test_path": test_path,
                    "temp_path": temp_path,
                    "defects": defects,
                })
    return records


def extract_patch(img: Image.Image, cx: int, cy: int, patch_size: int = 224) -> Image.Image:
    """Extrai patch de 224x224 centralizado em (cx, cy) com clamping nas bordas."""
    w, h = img.size
    half = patch_size // 2
    x1 = max(0, min(w - patch_size, cx - half))
    y1 = max(0, min(h - patch_size, cy - half))
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return img.crop((x1, y1, x2, y2)).convert("RGB")


def build_deeppcb_dataset(
    output_root: Path,
    train_samples_per_class: int = 200,
    val_samples_per_class: int = 40,
    sample_pool_size: int = 10,
    patch_size: int = 224,
    seed: int = 42,
) -> None:
    """Executa a extração estruturada de patches reais do DeepPCB."""
    random.seed(seed)
    pcb_data_dir = output_root / "data" / "deeppcb_raw" / "PCBData"
    if not pcb_data_dir.exists():
        raise FileNotFoundError(f"DeepPCB não encontrado em {pcb_data_dir}")

    print(f"[INFO] Lendo metadados e anotações de {pcb_data_dir}...")
    records = parse_annotations_and_images(pcb_data_dir)
    print(f"[SUCCESS] {len(records)} pares de imagens carregados.")

    # Group-Aware Split: 80% grupos/imagens para treino, 20% para validação
    random.shuffle(records)
    split_idx = int(len(records) * 0.8)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    raw_dir = output_root / "data" / "raw"
    pool_dir = output_root / "data" / "sample_pool"

    # Limpeza segura prévia
    for path in [raw_dir / "train", raw_dir / "val", pool_dir]:
        if path.exists():
            shutil.rmtree(path)

    classes_map = {
        "normal": "NORMAL",
        "defect_short": "DEFECT_SHORT",       # Tipo 2
        "defect_open": "DEFECT_OPEN",         # Tipo 1
        "defect_missing_hole": "DEFECT_MISSING_HOLE", # Tipo 6 (pin-hole)
        "defect_spurious": "DEFECT_SPURIOUS", # Tipos 4 (spur) e 5 (copper)
    }

    type_to_cls = {
        1: "defect_open",
        2: "defect_short",
        3: "unknown",
        4: "defect_spurious",
        5: "defect_spurious",
        6: "defect_missing_hole",
    }

    # Estruturar diretórios
    for split in ["train", "val"]:
        for c in classes_map:
            (raw_dir / split / c).mkdir(parents=True, exist_ok=True)
    for c in list(classes_map.keys()) + ["unknown"]:
        (pool_dir / c).mkdir(parents=True, exist_ok=True)

    print("[INFO] Extraindo patches reais de alta fidelidade...")

    for split_name, rec_list, target_count in [
        ("train", train_records, train_samples_per_class),
        ("val", val_records, val_samples_per_class),
    ]:
        counts = {c: 0 for c in classes_map}

        # 1. Extração de Defeitos
        for rec in rec_list:
            if all(counts[c] >= target_count for c in counts):
                break

            test_img = None
            for d in rec["defects"]:
                cls_folder = type_to_cls.get(d["type"])
                if cls_folder in counts and counts[cls_folder] < target_count:
                    if test_img is None:
                        test_img = Image.open(rec["test_path"])
                    cx = (d["x1"] + d["x2"]) // 2
                    cy = (d["y1"] + d["y2"]) // 2
                    patch = extract_patch(test_img, cx, cy, patch_size=patch_size)

                    save_path = raw_dir / split_name / cls_folder / f"{cls_folder}_{counts[cls_folder]:04d}.png"
                    patch.save(save_path)
                    counts[cls_folder] += 1

        # 2. Extração de Normais (Template perfeito sem defeito)
        normal_count = counts["normal"]
        for rec in rec_list:
            if normal_count >= target_count:
                break
            temp_img = Image.open(rec["temp_path"])
            # Extrair múltiplos crops não-sobrepostos do template
            w, h = temp_img.size
            grid_positions = [
                (112, 112), (320, 112), (528, 112),
                (112, 320), (320, 320), (528, 320),
                (112, 528), (320, 528), (528, 528),
            ]
            random.shuffle(grid_positions)
            for gx, gy in grid_positions:
                if normal_count >= target_count:
                    break
                patch = extract_patch(temp_img, gx, gy, patch_size=patch_size)
                save_path = raw_dir / split_name / "normal" / f"normal_{normal_count:04d}.png"
                patch.save(save_path)
                normal_count += 1
        counts["normal"] = normal_count

        print(f"[SUCCESS] Split '{split_name}' concluído: {counts}")

    # 3. População do Sample Pool da Demonstração (com dados reais separados de val)
    pool_counts = {c: 0 for c in list(classes_map.keys()) + ["unknown"]}
    val_records_pool = list(reversed(val_records))  # pega do final para não conflitar com val inicial

    for rec in val_records_pool:
        if all(pool_counts[c] >= sample_pool_size for c in pool_counts):
            break
        # Defeitos
        test_img = None
        for d in rec["defects"]:
            cls_folder = type_to_cls.get(d["type"])
            if cls_folder in pool_counts and pool_counts[cls_folder] < sample_pool_size:
                if test_img is None:
                    test_img = Image.open(rec["test_path"])
                cx = (d["x1"] + d["x2"]) // 2
                cy = (d["y1"] + d["y2"]) // 2
                patch = extract_patch(test_img, cx, cy, patch_size=patch_size)
                patch.save(pool_dir / cls_folder / f"sample_{cls_folder}_{pool_counts[cls_folder]+1:02d}.png")
                pool_counts[cls_folder] += 1

        # Normais
        if pool_counts["normal"] < sample_pool_size:
            temp_img = Image.open(rec["temp_path"])
            patch = extract_patch(temp_img, 320, 320, patch_size=patch_size)
            patch.save(pool_dir / "normal" / f"sample_normal_{pool_counts['normal']+1:02d}.png")
            pool_counts["normal"] += 1

    print(f"[SUCCESS] Sample Pool povoado com amostras reais da fábrica: {pool_counts}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    build_deeppcb_dataset(
        output_root=project_root,
        train_samples_per_class=200,
        val_samples_per_class=40,
        sample_pool_size=10,
        patch_size=224,
    )
