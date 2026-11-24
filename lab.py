import argparse
import csv
import hashlib
import json
import math
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

MODELS = {
    "clip": ("openai/clip-vit-base-patch32", "CLIPVisionModelWithProjection",
             (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "siglip": ("google/siglip-base-patch16-224", "SiglipVisionModel",
               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "dinov2": ("facebook/dinov2-base", "Dinov2Model",
               (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_photos(root, split):
    manifest = root / (split + ".csv")
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "person_id"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest}: required columns are path,person_id")
        rows = [(root / row["path"].strip(), row["person_id"].strip()) for row in reader]
    if not rows or any(not subject or not path.is_file() for path, subject in rows):
        raise ValueError(f"{manifest}: missing rows, person IDs or image files")
    return rows


def dataset_layout(root):
    train, test = (read_photos(root, split) for split in ("train", "test"))
    shared = {subject for _, subject in train} & {subject for _, subject in test}
    if shared:
        raise ValueError(f"Person IDs occur in both splits: {sorted(shared)[:5]}")
    references = {}
    for split in ("train", "test"):
        for kind in ("target", "neutral"):
            folder = root / "references" / kind / split
            paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"No reference images in {folder}")
            references[split, kind] = paths
    pools = [[path for path, _ in train], [path for path, _ in test], *references.values()]
    seen = set()
    for pool in pools:
        hashes = [digest(path) for path in pool]
        if len(hashes) != len(set(hashes)) or seen.intersection(hashes):
            raise ValueError("Identical image files found within or across data groups")
        seen.update(hashes)
    return train, test, references


def to_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255


def to_image(tensor):
    colors = tensor.detach().clamp(0, 1).mul(255).round()
    return Image.fromarray(colors[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))


def square_batch(images, size):
    height, width = images.shape[-2:]
    factor = size / max(height, width)
    new_height, new_width = (max(1, round(axis * factor)) for axis in (height, width))
    resized = F.interpolate(images, (new_height, new_width), mode="bilinear",
                            align_corners=False, antialias=True)
    vertical, horizontal = size - new_height, size - new_width
    return F.pad(resized, (horizontal // 2, horizontal - horizontal // 2,
                           vertical // 2, vertical - vertical // 2), mode="replicate")


class Portraits(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = read_image(self.rows[index][0])
        return square_batch(to_tensor(image)[None], self.size)[0].mul(255).round().div(255)


class Proxy(nn.Module):
    def __init__(self, name):
        super().__init__()
        import transformers

        identifier, model_class, mean, std = MODELS[name]
        self.name = name
        self.encoder = getattr(transformers, model_class).from_pretrained(identifier)
        self.encoder.eval().requires_grad_(False)
        self.size = 224 if name == "dinov2" else self.encoder.config.image_size
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def forward(self, images):
        pixels = (square_batch(images, self.size) - self.mean) / self.std
        outputs = self.encoder(pixel_values=pixels)
        features = {
            "clip": lambda: outputs.image_embeds,
            "siglip": lambda: outputs.pooler_output,
            "dinov2": lambda: outputs.last_hidden_state[:, 0],
        }[self.name]()
        return F.normalize(features, dim=-1)


class NoiseNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 3, 3, padding=1), nn.Tanh(),
        )
        nn.init.normal_(self[-2].weight, std=0.01)
        nn.init.zeros_(self[-2].bias)


def protect(model, images, eps):
    continuous = (images + model(images) * (eps / 255)).clamp(0, 1)
    return quantize(continuous)


def quantize(continuous):
    quantized = (continuous * 255).round() / 255
    # Score the actual 8-bit export in forward, and pass gradients through rounding.
    return quantized.detach() + (continuous - continuous.detach())


@torch.enable_grad()
def refine(original, protected, scorers, steps, eps, rng):
    bound = eps / 255
    delta = protected - original
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        candidate = quantize((original + delta).clamp(0, 1))
        view = training_view(candidate, rng.randrange(3))
        scores = [torch.mean(proxy(view) @ target) for proxy, target in scorers.values()]
        gradient, = torch.autograd.grad(torch.stack(scores).mean(), delta)
        with torch.no_grad():
            delta = (delta + gradient.sign() / 255).clamp(-bound, bound)
            delta = (original + delta).clamp(0, 1) - original
    return quantize((original + delta).clamp(0, 1))


@torch.no_grad()
def prototype(proxy, paths, device):
    vectors = [proxy(to_tensor(read_image(path))[None].to(device))[0] for path in paths]
    return F.normalize(torch.stack(vectors).mean(dim=0), dim=0)


def scorers(names, references, split, device):
    result = {}
    for name in names:
        proxy = Proxy(name).to(device)
        target = prototype(proxy, references[split, "target"], device)
        neutral = prototype(proxy, references[split, "neutral"], device)
        result[name] = proxy, target - neutral
    return result


def jpeg(images, quality):
    results = []
    for sample in images:
        buffer = BytesIO()
        to_image(sample[None]).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            results.append(to_tensor(decoded.convert("RGB")))
    return torch.stack(results).to(images.device)


def training_view(images, mode):
    if mode == 1:
        return F.interpolate(images, scale_factor=0.75, mode="bilinear",
                             align_corners=False, antialias=True)
    if mode == 2:
        compressed = jpeg(images, 85)
        return compressed + (images - images.detach())
    return images


def resolve_device(arg):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def train(args):
    rows, _, refs = dataset_layout(args.data)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.out.exists():
        raise ValueError(f"Checkpoint already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = NoiseNet().to(device)
    proxies = scorers(args.models, refs, "train", device)
    batches = DataLoader(Portraits(rows, args.size), batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    budget = args.eps / 255
    for epoch in range(args.epochs):
        values = []
        for original in batches:
            original = original.to(device)
            protected = protect(model, original, args.eps)
            view = training_view(protected, random.randrange(3))
            optimizer.zero_grad(set_to_none=True)
            scores = [torch.mean(proxy(view) @ target) for proxy, target in proxies.values()]
            mse = (protected - original).square().mean() / (budget ** 2)
            loss = -torch.stack(scores).mean() + args.mse_weight * mse
            loss.backward()
            optimizer.step()
            values.append([loss.detach().item(), mse.detach().item()])
        print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(values, axis=0)[0]),
                          "relative_mse": float(np.mean(values, axis=0)[1])}), flush=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save({"version": 1, "model": model.state_dict(), "eps": args.eps,
                "models": list(args.models), "seed": args.seed, "size": args.size,
                "library_versions": {"torch": str(torch.__version__)}}, temp)
    temp.replace(args.out)


def shuffled_delta(original, protected, eps, seed):
    delta = (protected - original).mul(255).round().to(torch.int16)
    order = torch.randperm(delta.shape[-2] * delta.shape[-1], generator=torch.Generator().manual_seed(seed))
    shuffled = delta.flatten(2)[:, :, order].reshape_as(delta).to(original.device)
    return (original + shuffled.clamp(-eps, eps) / 255).clamp(0, 1)


def quality_metrics(original, image):
    difference = (image - original).mul(255).round()
    mse = difference.square().mean().item()
    return {
        "max_channel_delta": int(difference.abs().max().item()),
        "mean_channel_abs_delta": round(difference.abs().mean().item(), 5),
        # Identical pixels have infinite PSNR. A blank field represents that case.
        "psnr_db": round(10 * math.log10(255 ** 2 / mse), 5) if mse else None,
    }


@torch.no_grad()
def evaluate(args):
    _, rows, refs = dataset_layout(args.data)
    if args.out.exists():
        raise ValueError(f"Evaluation folder already exists: {args.out}")
    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    models = saved["models"]
    if args.holdout in models:
        raise ValueError("Holdout model must not be used in training")
    model = NoiseNet().to(device).eval()
    model.load_state_dict(saved["model"])
    proxies = scorers([*models, args.holdout], refs, "test", device)
    training_scorers = {}
    if args.refine_steps:
        for name in models:
            proxy = proxies[name][0]
            target = prototype(proxy, refs["train", "target"], device)
            neutral = prototype(proxy, refs["train", "neutral"], device)
            training_scorers[name] = proxy, target - neutral
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "blind").mkdir()
    (args.out / "images").mkdir()
    table, pairs, blind, blind_key = [], [], [], []
    randomizer = random.Random(args.seed)
    refine_rng = random.Random(args.seed)
    for index, (path, _) in enumerate(rows, 1):
        sample_id = f"{index:04d}"
        original = to_tensor(read_image(path))[None].to(device)
        protected = protect(model, original, saved["eps"])
        if args.refine_steps:
            protected = refine(original, protected, training_scorers, args.refine_steps,
                               saved["eps"], refine_rng)
        images = {"clean": original, "shuffled": shuffled_delta(original, protected, saved["eps"], args.seed + index),
                  "protected": protected}
        pair = {"sample_id": sample_id}
        for variant, image in images.items():
            filename = f"images/{sample_id}-{variant}.png"
            to_image(image).save(args.out / filename)
            # Metrics and downstream API see the same exported pixels.
            exported = to_tensor(read_image(args.out / filename))[None].to(device)
            images[variant] = exported
            pair[variant] = filename
            metrics = quality_metrics(original, exported)
            if metrics["max_channel_delta"] > saved["eps"]:
                raise RuntimeError(f"Pixel limit exceeded for {filename}")
            compressed = jpeg(exported, args.jpeg_quality)
            for name, (proxy, target) in proxies.items():
                table.append({
                    "sample_id": sample_id, "variant": variant, "model": name,
                    "role": "holdout" if name == args.holdout else "training_proxy",
                    **metrics,
                    "raw_target_minus_neutral": round((proxy(exported) @ target).item(), 6),
                    "jpeg_target_minus_neutral": round((proxy(compressed) @ target).item(), 6),
                })
        pairs.append(pair)
        order = ["clean", "protected"]
        randomizer.shuffle(order)
        for label, variant in zip(("A", "B"), order):
            to_image(images[variant]).save(args.out / f"blind/{sample_id}-{label}.png")
        blind.append({"sample_id": sample_id, "distorted_image": "", "notes": ""})
        blind_key.append({"sample_id": sample_id, "protected_image": "AB"[order.index("protected")]})
    save_csv(args.out / "local.csv", table)
    save_csv(args.out / "blind-review.csv", blind)
    save_csv(args.out / "blind-key.csv", blind_key)
    save_json(args.out / "pairs.json", pairs)
    scores = {}
    for view in ("raw_target_minus_neutral", "jpeg_target_minus_neutral"):
        view_rows = {(r["sample_id"], r["variant"]): r[view] for r in table if r["role"] == "holdout"}
        scores[view] = {
            f"protected_minus_{baseline}": round(float(np.mean([
                view_rows[p["sample_id"], "protected"] - view_rows[p["sample_id"], baseline]
                for p in pairs
            ])), 6)
            for baseline in ("clean", "shuffled")
        }
    changed = sum(r["max_channel_delta"] > 0 for r in table
                  if r["role"] == "holdout" and r["variant"] == "protected")
    summary = {"samples": len(rows), "nonzero_protected_samples": changed,
               "epsilon_8bit": saved["eps"], "holdout": args.holdout,
               "jpeg_quality": args.jpeg_quality, "refine_steps": args.refine_steps,
               "holdout_mean_deltas": scores}
    save_json(args.out / "local-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Learn and evaluate copyright-targeted pixel perturbations")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    for dest in (training,):
        dest.add_argument("--data", required=True, type=Path)
        dest.add_argument("--device", default="auto")
        dest.add_argument("--seed", type=int, default=42)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--models", choices=tuple(MODELS), nargs="+", default=["clip", "siglip"])
    training.add_argument("--eps", type=int, choices=range(1, 33), default=4, metavar="1..32")
    training.add_argument("--epochs", type=int, default=8)
    training.add_argument("--batch", type=int, default=2)
    training.add_argument("--size", type=int, default=256)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--mse-weight", type=float, default=0.05)
    testing = commands.add_parser("evaluate")
    testing.add_argument("--data", required=True, type=Path)
    testing.add_argument("--device", default="auto")
    testing.add_argument("--seed", type=int, default=42)
    testing.add_argument("--checkpoint", required=True, type=Path)
    testing.add_argument("--out", required=True, type=Path)
    testing.add_argument("--holdout", choices=tuple(MODELS), default="dinov2")
    testing.add_argument("--refine-steps", type=int, default=0,
                         help="Optional per-photo gradient refinement using training proxies and references")
    testing.add_argument("--jpeg-quality", type=int, choices=range(1, 101), default=85, metavar="1..100")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.epochs, args.batch) <= 0 or args.size < 32 or args.lr <= 0 or args.mse_weight < 0:
            parser.error("Positive training settings, size >= 32 and nonnegative MSE weight are required")
        if len(set(args.models)) != len(args.models):
            parser.error("Training proxies must be distinct")
        if set(args.models) == set(MODELS):
            parser.error("Leave at least one model out of training for evaluation")
        train(args)
    else:
        if args.refine_steps < 0:
            parser.error("--refine-steps must be nonnegative")
        evaluate(args)


if __name__ == "__main__":
    main()
