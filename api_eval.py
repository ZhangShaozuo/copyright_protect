import argparse
import base64
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

VARIANTS = ("clean", "shuffled", "protected")
POLICY_CODES = {"moderation_blocked", "content_policy_violation", "image_content_policy_violation"}


def digest(path):
    with path.open("rb") as image:
        return hashlib.file_digest(image, "sha256").hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_error(error):
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    if not isinstance(details, dict):
        details = {}
    code = details.get("code") or error.code
    return {
        "outcome": "policy_refusal" if code in POLICY_CODES else "api_error",
        "status": error.status_code,
        "error_code": code,
        "message": details.get("message") or str(error),
        "moderation_details": details.get("moderation_details"),
        "request_id": error.request_id,
    }


def summarize(rows):
    trials, controls = {}, []
    for row in rows:
        if row["variant"].endswith("_control"):
            controls.append({k: row.get(k) for k in ("sample_id", "variant", "outcome", "moderation_details")})
        else:
            trials.setdefault((row["sample_id"], row["repeat"]), {})[row["variant"]] = row
    conclusive = [
        variants for variants in trials.values()
        if set(variants) == set(VARIANTS)
        and variants["clean"]["outcome"] == "image_returned"
        and all(r["outcome"] in {"image_returned", "policy_refusal"} for r in variants.values())
    ]
    attribution = {}
    for row in rows:
        if row["outcome"] == "policy_refusal":
            details = row.get("moderation_details") or {}
            attribution[row["variant"]] = attribution.get(row["variant"], []) + [details]
    return {
        "paired_trials": len(trials), "conclusive_trials_where_clean_produced": len(conclusive),
        "excluded_trials": len(trials) - len(conclusive),
        "protected_refusals": sum(p["protected"]["outcome"] == "policy_refusal" for p in conclusive),
        "shuffled_refusals": sum(p["shuffled"]["outcome"] == "policy_refusal" for p in conclusive),
        "refusal_details": attribution, "controls": controls,
    }


def run(args):
    pairs = json.loads((args.eval / "pairs.json").read_text(encoding="utf-8")) if args.eval else []
    if args.require_blind_pass:
        with (args.eval / "blind-review.csv").open(encoding="utf-8-sig", newline="") as file:
            passed = {row["sample_id"] for row in csv.DictReader(file)
                      if row["distorted_image"].strip().lower() == "none"}
        pairs = [pair for pair in pairs if pair["sample_id"] in passed]
    pairs = pairs[:args.limit]
    if args.eval and not pairs:
        raise ValueError("No evaluation pairs found")
    jobs = [
        (pair["sample_id"], repeat, variant, args.eval / pair[variant])
        for pair in pairs for repeat in range(1, args.repeats + 1) for variant in VARIANTS
    ]
    random.Random(args.seed).shuffle(jobs)
    controls = [
        (f"{kind}{index:02d}", 1, kind + "_control", path)
        for kind, paths in (("target", args.target_control), ("clean", args.clean_control))
        for index, path in enumerate(paths, 1)
    ]
    jobs = controls + jobs
    plan = [{"sample_id": sample, "repeat": repeat, "variant": variant, "sha256": digest(path)}
            for sample, repeat, variant, path in jobs]
    settings = {"model": args.model, "prompt": args.prompt, "jobs": plan}
    if not args.submit:
        print(json.dumps({"requests": len(jobs), "model": args.model, "submitted": False}))
        return

    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY before using --submit")
    client = openai.OpenAI(api_key=key, base_url="https://api.openai.com/v1",
                           max_retries=0, timeout=240)
    if not args.out.exists():
        args.out.mkdir(parents=True)
        save_json(args.out / "plan.json", settings)
    if json.loads((args.out / "plan.json").read_text(encoding="utf-8")) != settings:
        raise ValueError("Output folder is bound to a different request plan")
    rows = []
    for metadata, (sample, repeat, variant, path) in zip(plan, jobs):
        stem = f"{sample}-{variant}-{repeat}"
        report_path = args.out / (stem + ".json")
        if report_path.exists():
            rows.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        if digest(path) != metadata["sha256"]:
            raise ValueError(f"Input changed after planning: {path}")
        record = {**metadata, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        try:
            with path.open("rb") as file:
                response = client.images.edit(model=args.model, prompt=args.prompt, image=file, n=1)
        except openai.APIStatusError as error:
            record.update(normalized_error(error))
        except openai.APIConnectionError as error:
            record.update({"outcome": "connection_error", "message": str(error)})
        else:
            payload = response.data[0].b64_json if response.data else None
            if payload:
                image_path = args.out / (stem + ".png")
                image_path.write_bytes(base64.b64decode(payload))
                record.update({"outcome": "image_returned", "output": image_path.name,
                               "request_id": response._request_id})
            else:
                record.update({"outcome": "no_image_no_policy_code", "request_id": response._request_id})
        save_json(report_path, record)
        rows.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record.get("status") in (401, 403, 429) or record["outcome"] == "connection_error":
            break
    result = {"requested": len(jobs), "recorded": len(rows), **summarize(rows)}
    save_json(args.out / "api-summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate local sample pairs on the OpenAI Image API")
    parser.add_argument("--eval", type=Path, help="Folder produced by lab.py evaluate")
    parser.add_argument("--out", required=True, type=Path, help="API output and logs")
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--prompt", default="Put this person inside an office")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-control", type=Path, nargs="*", default=[])
    parser.add_argument("--clean-control", type=Path, nargs="*", default=[])
    parser.add_argument("--require-blind-pass", action="store_true",
                        help="Select only pairs marked 'none' in blind-review.csv")
    parser.add_argument("--submit", action="store_true", help="Run paid cloud requests")
    args = parser.parse_args()
    if args.limit <= 0 or args.repeats <= 0:
        parser.error("--limit and --repeats must be positive")
    if args.require_blind_pass and not args.eval:
        parser.error("--require-blind-pass needs --eval")
    if not args.eval and not (args.target_control and args.clean_control):
        parser.error("Provide --eval or both --target-control and --clean-control")
    run(args)
