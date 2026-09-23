"""Gerador Industrial de Benchmarks de PCBs (Placas de Circuito Impresso).

Sintetiza amostras industriais fotorrealistas de alta fidelidade para treinamento,
avaliação e demonstração ao vivo, cobrindo:
1. NORMAL (Placa conforme)
2. DEFECT_SHORT (Curto-circuito por ponte de solda)
3. DEFECT_OPEN (Trilha rompida / circuito aberto)
4. DEFECT_MISSING_HOLE (Furo de via/ilhó ausente)
5. DEFECT_SPURIOUS (Cobre/estanho espúrio)
6. UNKNOWN_ANOMALY (Risco físico/arranhão no substrato para Open-Set test)
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Força UTF-8 no stdout para compatibilidade com consoles Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configurações de cores industriais padrão FR-4
FR4_GREENS = [
    (15, 65, 35),
    (20, 75, 40),
    (10, 55, 30),
]
COPPER_TRACES = [
    (185, 140, 50),
    (210, 160, 60),
    (170, 125, 40),
]
SOLDER_SILVER = [
    (200, 205, 215),
    (220, 225, 230),
    (180, 185, 195),
]
HOLE_BLACK = (15, 15, 15)
SILK_WHITE = (235, 240, 245)


def create_base_pcb(size: int = 224, seed: int | None = None) -> tuple[Image.Image, list[dict]]:
    """Gera a geometria base de uma PCB com barramentos, trilhas e ilhas."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Substrato FR-4 com textura de fibra de vidro
    base_color = random.choice(FR4_GREENS)
    img_arr = np.full((size, size, 3), base_color, dtype=np.uint8)
    noise = np.random.normal(0, 3, (size, size, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    traces = []
    # Barramentos horizontais e diagonais
    num_bus = random.randint(4, 7)
    y_spacing = size // (num_bus + 1)

    for i in range(num_bus):
        y = (i + 1) * y_spacing + random.randint(-4, 4)
        trace_color = random.choice(COPPER_TRACES)
        width = random.choice([3, 4, 5])

        # Padrão com quebra em 45 graus típico de roteamento PCB
        x_turn1 = random.randint(size // 4, size // 3)
        x_turn2 = x_turn1 + random.randint(20, 40)
        y_offset = random.choice([-15, 15, -20, 20])

        points = [
            (0, y),
            (x_turn1, y),
            (x_turn2, y + y_offset),
            (size, y + y_offset),
        ]
        draw.line(points, fill=trace_color, width=width)
        traces.append({"points": points, "color": trace_color, "width": width})

    # Ilhas de solda (Pads) e vias de passagem com furos
    pads = []
    num_pads = random.randint(8, 14)
    for _ in range(num_pads):
        px = random.randint(25, size - 25)
        py = random.randint(25, size - 25)
        rad = random.randint(7, 11)
        pad_color = random.choice(SOLDER_SILVER)

        # Pad circular de solda
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=pad_color, outline=random.choice(COPPER_TRACES), width=2)
        # Furo de passagem da broca (via/through-hole)
        hole_rad = rad // 2
        draw.ellipse([px - hole_rad, py - hole_rad, px + hole_rad, py + hole_rad], fill=HOLE_BLACK)
        pads.append({"x": px, "y": py, "rad": rad, "hole_rad": hole_rad, "color": pad_color})

    # Textos de serigrafia fabril (Silkscreen)
    labels = ["POSI-SMT", "C12", "R4", "U1", "GND", "+5V", "REV_B"]
    for _ in range(random.randint(2, 4)):
        lx = random.randint(15, size - 50)
        ly = random.randint(15, size - 25)
        text = random.choice(labels)
        draw.text((lx, ly), text, fill=SILK_WHITE)

    return img, {"traces": traces, "pads": pads}


def inject_defect(
    base_img: Image.Image,
    meta: dict,
    defect_type: str,
    size: int = 224,
) -> tuple[Image.Image, dict]:
    """Injeta defeito físico controlado na geometria da PCB."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    info = {"defect_type": defect_type, "bbox": None}

    if defect_type == "NORMAL":
        return img, info

    elif defect_type == "DEFECT_SHORT":
        # Curto-circuito: ponte de solda/estanho entre dois barramentos próximos
        traces = meta.get("traces", [])
        if len(traces) >= 2:
            t1 = traces[0]["points"]
            t2 = traces[1]["points"]
            # Coordenada intermediária
            x = random.randint(size // 3, 2 * size // 3)
            y1 = t1[1][1]
            y2 = t2[1][1]
            draw.line([(x, y1), (x + random.randint(-4, 4), y2)], fill=(225, 230, 235), width=random.randint(4, 7))
            info["bbox"] = [min(x - 5, x - 5), min(y1, y2), max(x + 5, x + 5), max(y1, y2)]
        else:
            x, y = size // 2, size // 2
            draw.line([(x, y - 15), (x, y + 15)], fill=(225, 230, 235), width=5)
            info["bbox"] = [x - 5, y - 15, x + 5, y + 15]

    elif defect_type == "DEFECT_OPEN":
        # Trilha rompida: gap do substrato apagando pedaço da trilha
        traces = meta.get("traces", [])
        if traces:
            t = random.choice(traces)["points"]
            pt = t[1]
            x, y = pt[0] + 10, pt[1]
        else:
            x, y = size // 2, size // 2
        bg_color = FR4_GREENS[0]
        # Corta a trilha com um retângulo da cor do substrato
        gap_w = random.randint(10, 16)
        draw.rectangle([x - gap_w // 2, y - 8, x + gap_w // 2, y + 8], fill=bg_color)
        info["bbox"] = [x - gap_w // 2, y - 8, x + gap_w // 2, y + 8]

    elif defect_type == "DEFECT_MISSING_HOLE":
        # Furo ausente: o pad existe, mas a broca não furou (sem furo preto)
        pads = meta.get("pads", [])
        if pads:
            pad = random.choice(pads)
            px, py, rad = pad["x"], pad["y"], pad["rad"]
            # Preenche o furo com solda completa
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=pad["color"])
            info["bbox"] = [px - rad, py - rad, px + rad, py + rad]
        else:
            px, py = size // 3, size // 3
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(210, 215, 220))
            info["bbox"] = [px - 8, py - 8, px + 8, py + 8]

    elif defect_type == "DEFECT_SPURIOUS":
        # Cobre/solda espúria: gota condutora aleatória no substrato isolante
        pads = meta.get("pads", [])
        px = random.randint(30, size - 30)
        py = random.randint(30, size - 30)
        # Desenha respingo metálico irregular
        rad = random.randint(5, 9)
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(215, 170, 60))
        draw.ellipse([px - 2, py - rad - 3, px + 3, py], fill=(215, 170, 60))
        info["bbox"] = [px - rad, py - rad - 3, px + rad, py + rad]

    elif defect_type == "UNKNOWN_ANOMALY":
        # Anomalia inédita (Open-Set): arranhão profundo / queimadura transversal
        x1 = random.randint(10, size // 4)
        y1 = random.randint(10, size // 4)
        x2 = random.randint(3 * size // 4, size - 10)
        y2 = random.randint(3 * size // 4, size - 10)
        # Risco mecânico violento cruzando várias trilhas
        draw.line([(x1, y1), (x2, y2)], fill=(40, 20, 10), width=3)
        draw.line([(x1 + 1, y1), (x2 + 1, y2)], fill=(120, 70, 30), width=1)
        info["bbox"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    return img, info


def generate_dataset(
    output_dir: Path,
    samples_per_class: int = 120,
    sample_pool_size: int = 8,
    img_size: int = 224,
) -> None:
    """Gera o dataset completo de treino/validação e o pool da Live Demo."""
    classes = [
        "NORMAL",
        "DEFECT_SHORT",
        "DEFECT_OPEN",
        "DEFECT_MISSING_HOLE",
        "DEFECT_SPURIOUS",
    ]
    raw_dir = output_dir / "data" / "raw"
    pool_dir = output_dir / "data" / "sample_pool"

    print("[INFO] Iniciando geracao de dados sinteticos industriais de alta fidelidade...")
    total_imgs = 0

    for cls_name in classes:
        cls_lower = cls_name.lower()
        train_path = raw_dir / "train" / cls_lower
        val_path = raw_dir / "val" / cls_lower
        demo_pool_path = pool_dir / cls_lower

        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        demo_pool_path.mkdir(parents=True, exist_ok=True)

        n_train = int(samples_per_class * 0.8)
        n_val = samples_per_class - n_train

        # 1. Gerar Treino
        for idx in range(n_train):
            base, meta = create_base_pcb(size=img_size)
            defective_img, _ = inject_defect(base, meta, cls_name, size=img_size)
            defective_img.save(train_path / f"{cls_lower}_{idx:04d}.png")
            total_imgs += 1

        # 2. Gerar Validação
        for idx in range(n_val):
            base, meta = create_base_pcb(size=img_size)
            defective_img, _ = inject_defect(base, meta, cls_name, size=img_size)
            defective_img.save(val_path / f"{cls_lower}_val_{idx:04d}.png")
            total_imgs += 1

        # 3. Gerar Pool da Live Demo
        for idx in range(sample_pool_size):
            base, meta = create_base_pcb(size=img_size)
            defective_img, _ = inject_defect(base, meta, cls_name, size=img_size)
            defective_img.save(demo_pool_path / f"sample_{cls_lower}_{idx + 1:02d}.png")

    # 4. Gerar Amostras de Anomalia Inédita (Open-Set Anomaly) para o Sample Pool
    unknown_pool_path = pool_dir / "unknown"
    unknown_pool_path.mkdir(parents=True, exist_ok=True)
    for idx in range(sample_pool_size):
        base, meta = create_base_pcb(size=img_size)
        unknown_img, _ = inject_defect(base, meta, "UNKNOWN_ANOMALY", size=img_size)
        unknown_img.save(unknown_pool_path / f"sample_unknown_{idx + 1:02d}.png")

    print(f"[SUCCESS] Geracao concluida com sucesso! Total de {total_imgs} imagens geradas.")
    print(f"[SUCCESS] Pool da Live Demo populado em '{pool_dir}' com classes industriais e anomalias.")


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    generate_dataset(output_dir=root_path, samples_per_class=100, sample_pool_size=10, img_size=224)
