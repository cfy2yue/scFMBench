#!/usr/bin/env python3
"""Generate the Nature-style benchmark figure set under ``output/figures/``.

Usage (from scFM root):
    python benchmark/cli/build_figures.py [--scfm-root .] [--out-dir output/figures]
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Callable

import sys

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "fm"))

from benchmark.plot import data as D
from benchmark.plot import figures as F
from benchmark.plot import style as ST
import paths


def _try_figure(
    name: str,
    builder: Callable[[], tuple[Path, Path]],
) -> tuple[dict[str, str], dict[str, str] | None]:
    try:
        pdf, png = builder()
    except Exception as exc:
        return {}, {
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    return {"name": name, "pdf": str(pdf), "png": str(png)}, None


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

    has_atlas = df["category"].isin(("atlas", "atlas_TS")).any()
    has_chempert = df["category"].eq("chempert").any()
    has_genepert = df["category"].eq("genepert").any()
    has_atlas_efficiency = (
        (paths.output_root() / "embeddings").glob("*/*/raw/meta.json")
    )
    has_atlas_efficiency = any(
        p.parents[1].name in {
            "Blood",
            "BoneMarrow",
            "Heart",
            "Lung",
            "LymphNode",
            "Skin",
            "TS_Immune_xtissue",
        }
        for p in has_atlas_efficiency
    )

    figure_specs = [
        ("fig1_overview", lambda: F.fig1_overview(df, out_dir), True, ""),
        ("fig2_atlas", lambda: F.fig2_atlas(df, out_dir), has_atlas, "no atlas metrics in summary_all.csv"),
        ("fig3_geometry", lambda: F.fig3_geometry(df, out_dir), True, ""),
        ("fig4_chempert", lambda: F.fig4_chempert(df, out_dir, per_pert_df=per_pert), has_chempert, "no chempert rows in summary_all.csv"),
        ("fig4b_genepert", lambda: F.fig4b_genepert(df, out_dir, per_pert_df=per_pert), has_genepert, "no genepert rows in summary_all.csv"),
        ("fig4_2_chempert_sim", lambda: F.fig4_2_chempert_sim(df, out_dir), has_chempert, "no chempert rows in summary_all.csv"),
        ("fig4b_2_genepert_sim", lambda: F.fig4b_2_genepert_sim(df, out_dir), has_genepert, "no genepert rows in summary_all.csv"),
        ("fig5_overall", lambda: F.fig5_overall(df, out_dir), True, ""),
        ("fig6_efficiency", lambda: F.fig6_efficiency(df, out_dir), has_atlas_efficiency, "no atlas throughput metadata under embeddings"),
        ("fig_supp_all_metrics", lambda: F.fig_supp_all_metrics(df, out_dir), True, ""),
    ]
    figure_records: list[dict[str, str]] = []
    failed_figures: list[dict[str, str]] = []
    skipped_figures: list[dict[str, str]] = []
    for name, builder, should_run, skip_reason in figure_specs:
        if not should_run:
            skipped_figures.append({"name": name, "reason": skip_reason})
            continue
        record, failure = _try_figure(name, builder)
        if failure:
            failed_figures.append(failure)
        else:
            figure_records.append(record)

    manifest = {
        "scfm_root": str(scfm),
        "out_dir": str(out_dir),
        "figures": figure_records,
        "failed_figures": failed_figures,
        "skipped_figures": skipped_figures,
        "n_figures": int(len(figure_records)),
        "n_failed_figures": int(len(failed_figures)),
        "n_skipped_figures": int(len(skipped_figures)),
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
