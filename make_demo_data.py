import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "photos").mkdir()
    generator = np.random.default_rng(11)
    for split, count in (("train", 6), ("test", 3)):
        rows = ["path,person_id"]
        for index in range(count):
            height, width = 107 + index * 4, 83 + index * 5
            image = Image.fromarray(generator.integers(15, 220, (height, width, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(image)
            draw.ellipse((18, 18, width - 14, height - 19), fill=(230, 173, 123))
            for x in (width // 3, width * 2 // 3):
                draw.ellipse((x - 2, 45, x + 3, 51), fill="black")
            draw.arc((width // 3, 58, width * 2 // 3, 73), 0, 180, fill="brown", width=2)
            name = f"{split}-{index}"
            image.save(root / "photos" / f"{name}.png")
            rows.append(f"photos/{name}.png,{name}")
        (root / f"{split}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        for kind in ("target", "neutral"):
            folder = root / "references" / kind / split
            folder.mkdir(parents=True)
            for index in range(3):
                image = Image.fromarray(generator.integers(40, 90, (111, 93, 3), dtype=np.uint8))
                draw = ImageDraw.Draw(image)
                if kind == "target":
                    draw.polygon([(46, 10 + index), (80, 55), (46, 100 - index), (12, 55)], fill=(95, 20, 225))
                else:
                    draw.ellipse((12 + index, 20, 81 - index, 90), fill=(80, 200, 220))
                image.save(folder / f"{index}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate geometric test images for a local smoke test")
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out)
