"""
thoravis/src/chexpert_label.py
─────────────────────────────────────────────────────────────────────────────
CheXpert Plus ships free-text radiology reports, not the classic structured
14-pathology labels (see src/chexpert_download.py). This derives them by
running CheXbert (Smit et al. 2020, EMNLP) — the standard BERT-based labeler
that produces the exact same 14 CheXpert categories from report text — on
the reports for whichever images were actually downloaded.

Deliberately has no import on the rest of this package (no `from src.x`):
CheXbert's own label.py is shelled out to as a separate process, so this
script never needs torch loaded in-process, and stays runnable even when
torch's DLL loading is flaky under memory pressure (see src/dataset.py).

Requires a local clone of https://github.com/stanfordmlgroup/CheXbert and
its checkpoint (official host is dead; a Stanford AIMI-hosted mirror is at
https://huggingface.co/StanfordAIMI/RRG_scorers/blob/main/chexbert.pth).

Usage
-----
    python src/chexpert_label.py \
        --reports-csv data/chexpert_valid/reports.csv \
        --image-root data/chexpert_valid/PNG_valid \
        --chexbert-repo <path to cloned CheXbert repo> \
        --checkpoint <path to chexbert.pth> \
        --out data/chexpert_valid/labeled_reports.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _resolve_image_path(image_root: Path, raw_path: str):
    """Same matching rule as src/chexpert_eval.py's resolver, duplicated
    (not imported) to keep this script torch-free — see module docstring."""
    p = Path(raw_path)
    candidates = [p, Path(*p.parts[1:])] if len(p.parts) > 1 else [p]
    for candidate in candidates:
        for ext in (candidate.suffix,) + _IMAGE_EXTENSIONS:
            target = image_root / candidate.with_suffix(ext)
            if target.exists():
                return target
    return None


def main():
    parser = argparse.ArgumentParser(description="Derive CheXpert-style labels from report text via CheXbert")
    parser.add_argument("--reports-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--chexbert-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="data/chexpert_valid/labeled_reports.csv")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    df = pd.read_csv(args.reports_csv)

    matched_rows = [i for i, row in df.iterrows() if _resolve_image_path(image_root, row["path_to_image"]) is not None]
    df = df.iloc[matched_rows].reset_index(drop=True)
    print(f"{len(df)} reports matched to a downloaded image under {image_root}.")
    if len(df) == 0:
        raise SystemExit("No reports matched any downloaded image — check --reports-csv / --image-root.")

    report_impression = df["section_impression"].fillna(df["report"])
    chexbert_input = pd.DataFrame({"Report Impression": report_impression})

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    input_csv = out_dir / "_chexbert_input.csv"
    chexbert_input.to_csv(input_csv, index=False)

    chexbert_src = Path(args.chexbert_repo) / "src"
    print("Running CheXbert (this loads BERT and runs inference over every report)...")
    subprocess.run(
        [sys.executable, "label.py", "-d", str(input_csv.resolve()), "-o", str(out_dir.resolve()), "-c", str(Path(args.checkpoint).resolve())],
        cwd=chexbert_src,
        check=True,
    )

    labeled = pd.read_csv(out_dir / "labeled_reports.csv")
    labeled.insert(0, "path_to_image", df["path_to_image"].values)
    labeled.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
