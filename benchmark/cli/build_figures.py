#!/usr/bin/env python3
"""Generate the Nature-style benchmark figure set under ``output/figures/``.

Usage (from scFM root):
    python benchmark/cli/build_figures.py [--scfm-root .] [--out-dir output/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "fm"))

from benchmark.plot import data as D
from benchmark.plot import figures as F
from benchmark.plot import style as ST
import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=_HERE.parents[2])
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <scfm-root>/output/figures")
    args = ap.parse_args()

    scfm = args.scfm_root.resolve()
    out_dir = (args.out_dir or paths.output_root() / "figures").resolve()

    ST.apply_rcparams()

    df = D.load_wide(scfm)
    per_pert = D.per_perturb_table(scfm)
    df = D.augment_with_topk_spearman(df, scfm, out_dir, per_pert)
    df = D.augment_with_mantel_spearman(df, scfm, out_dir)

    paths = []
    paths.append(F.fig1_overview(df, out_dir))
    paths.append(F.fig2_atlas(df, out_dir))
    paths.append(F.fig3_geometry(df, out_dir))
    paths.append(F.fig4_chempert(df, out_dir, per_pert_df=per_pert))
    paths.append(F.fig4b_genepert(df, out_dir, per_pert_df=per_pert))
    paths.append(F.fig4_2_chempert_sim(df, out_dir))
    paths.append(F.fig4b_2_genepert_sim(df, out_dir))
    paths.append(F.fig5_overall(df, out_dir))
    paths.append(F.fig6_efficiency(df, out_dir))
    paths.append(F.fig_supp_all_metrics(df, out_dir))

    manifest = {
        "scfm_root": str(scfm),
        "out_dir": str(out_dir),
        "figures": [{"pdf": str(pdf), "png": str(png)} for pdf, png in paths],
        "n_figures": int(len(paths)),
        "n_rows_summary_all": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_datasets": int(df["dataset_id"].nunique()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
