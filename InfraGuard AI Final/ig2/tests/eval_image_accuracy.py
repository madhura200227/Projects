"""
Manifest-driven evaluation for both image endpoints:
- /api/pothole/analyze logic (via PotholeDetector directly)
- /api/v1/predict/image logic (via InfraGuardModel.cnn directly)

Create a manifest at: ig2/data/eval/manifest.jsonl
One JSON object per line.

Supported fields (per item):
{
  "image_path": "ig2/data/samples/img1.jpg",

  "expect": {
    "infra_type": "road",
    "damage": {
      "crack": 10,
      "pothole": 80,
      "wear": 20,
      "water": 0,
      "deformation": 15
    },
    "potholes_detected": 2
  }
}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mae(a: float, b: float) -> float:
    return abs(a - b)


@dataclass
class ItemResult:
    image_path: str
    infra_type_ok: Optional[bool]
    pothole_count_ok: Optional[bool]
    damage_mae: Optional[float]
    details: Dict[str, Any]


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception as e:
            raise RuntimeError(f"Invalid JSON on line {i}: {e}")
    return items


def _eval_predict_image(img_path: str) -> Dict[str, Any]:
    import numpy as np
    from PIL import Image
    from ml.model import InfraGuardModel

    model = InfraGuardModel()
    model.load()

    pil = Image.open(img_path).convert("RGB")
    arr = np.array(pil)
    infra = model.cnn.classify_infra_type(arr)
    damage = model.cnn.predict_from_array(arr)
    return {"infra": infra, "damage": damage}


def _eval_pothole(img_path: str) -> Dict[str, Any]:
    from ml.pothole_detector import PotholeDetector

    det = PotholeDetector()
    det.load_models()
    out = det.analyze_image(img_path)
    return out


def evaluate(manifest_path: Path, mode: str) -> Tuple[List[ItemResult], Dict[str, Any]]:
    items = _load_manifest(manifest_path)
    results: List[ItemResult] = []

    infra_total = infra_ok = 0
    poth_total = poth_ok = 0
    damage_total = 0
    damage_mae_sum = 0.0

    for it in items:
        img = it["image_path"]
        exp = it.get("expect", {})
        details: Dict[str, Any] = {}

        infra_type_ok = None
        pothole_count_ok = None
        damage_mae = None

        if mode in ("predict_image", "both"):
            pred = _eval_predict_image(img)
            details["predict_image"] = pred

            exp_type = exp.get("infra_type")
            if exp_type is not None:
                infra_total += 1
                got_type = pred["infra"].get("detected_type")
                infra_type_ok = (got_type == exp_type)
                infra_ok += 1 if infra_type_ok else 0

            exp_damage = exp.get("damage")
            if exp_damage is not None:
                # Compare on the 0-100 scale
                keys = ["crack", "pothole", "wear", "water", "deformation"]
                diffs = []
                for k in keys:
                    if k in exp_damage:
                        diffs.append(_mae(float(pred["damage"].get(k, 0.0)), float(exp_damage[k])))
                if diffs:
                    damage_mae = sum(diffs) / len(diffs)
                    damage_total += 1
                    damage_mae_sum += damage_mae

        if mode in ("pothole", "both"):
            poth = _eval_pothole(img)
            details["pothole"] = poth

            exp_cnt = exp.get("potholes_detected")
            if exp_cnt is not None:
                poth_total += 1
                got_cnt = poth.get("potholes_detected", 0)
                pothole_count_ok = (int(got_cnt) == int(exp_cnt))
                poth_ok += 1 if pothole_count_ok else 0

        results.append(
            ItemResult(
                image_path=img,
                infra_type_ok=infra_type_ok,
                pothole_count_ok=pothole_count_ok,
                damage_mae=damage_mae,
                details=details,
            )
        )

    summary = {
        "count": len(results),
        "infra_type_accuracy": (infra_ok / infra_total) if infra_total else None,
        "pothole_count_accuracy": (poth_ok / poth_total) if poth_total else None,
        "damage_mae_avg": (damage_mae_sum / damage_total) if damage_total else None,
        "evaluated": {
            "infra_type_items": infra_total,
            "pothole_count_items": poth_total,
            "damage_items": damage_total,
        },
    }
    return results, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="ig2/data/eval/manifest.jsonl")
    ap.add_argument("--mode", choices=["predict_image", "pothole", "both"], default="both")
    ap.add_argument("--out", type=str, default="ig2/data/eval/report.json")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    res, summary = evaluate(manifest_path, args.mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "items": [
                    {
                        "image_path": r.image_path,
                        "infra_type_ok": r.infra_type_ok,
                        "pothole_count_ok": r.pothole_count_ok,
                        "damage_mae": r.damage_mae,
                        "details": r.details,
                    }
                    for r in res
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()

