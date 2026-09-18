"""
thoravis/src/chexpert_download.py
─────────────────────────────────────────────────────────────────────────────
One-time downloader for CheXpert Plus's validation split (234 images + labels)
via the Redivis API, for src/chexpert_eval.py's cross-dataset validation.

Requires:
  pip install redivis
  Your own approved Redivis access to Stanford AIMI's "CheXpert Plus" dataset
  (https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x).

The first run opens an interactive OAuth prompt (a URL printed to the
terminal) — no API token needed for a one-off pull like this. Approve it in
your own browser under your own Redivis account; the script resumes once
you do.

Usage
-----
    python src/chexpert_download.py --out data/chexpert_valid

Run as a plain script, not `-m src.chexpert_download` — `src/__init__.py`
eagerly imports torch (via src.dataset), which this script has no need for
and which can fail to load under memory pressure (see src/dataset.py and
the Docker section of README.md for the same issue elsewhere in this repo).
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download CheXpert Plus's validation split via Redivis")
    parser.add_argument("--out", default="data/chexpert_valid")
    args = parser.parse_args()

    import redivis  # deliberately imported late — most of the repo never needs this dependency

    out_dir = Path(args.out)
    images_dir = out_dir / "PNG_valid"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = redivis.organization("AIMI").dataset("chexpert_plus")

    print("Fetching metadata/reports table (df_chexpert_plus_240401)...")
    # NOTE: CheXpert Plus ships free-text reports, not the classic 14-column
    # structured pathology labels — those get derived separately by running
    # CheXbert on the report text (see src/chexpert_label.py).
    full_df = dataset.table("df_chexpert_plus_240401").to_pandas_dataframe()
    valid_df = full_df[full_df["split"] == "valid"].reset_index(drop=True)
    print(f"  {len(valid_df)} validation-split rows found (of {len(full_df)} total).")

    keep_cols = ["path_to_image", "section_impression", "report", "deid_patient_id", "frontal_lateral", "ap_pa", "split"]
    valid_df = valid_df[[c for c in keep_cols if c in valid_df.columns]]

    labels_csv = out_dir / "reports.csv"
    valid_df.to_csv(labels_csv, index=False)
    print(f"  Wrote {labels_csv}")

    print("Downloading PNG_valid image files...")
    png_valid = dataset.table("PNG_valid")
    n = 0
    for f in png_valid.list_files():
        target = images_dir / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f.read())
        n += 1
        if n % 25 == 0:
            print(f"  {n} files downloaded...")
    print(f"Done — {n} image files under {images_dir}")
    print(f"\nNext: label the reports with CheXbert (see src/chexpert_label.py), then run\n"
          f"  python -m src.chexpert_eval --labels-csv <labeled_reports.csv> --image-root {images_dir} --path-column path_to_image")


if __name__ == "__main__":
    main()
