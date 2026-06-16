"""TranscriptFormer expression-only adapter.

The official package exposes a CLI that writes cell embeddings to
``obsm["embeddings"]``.  This adapter writes the incoming AnnData to a temporary
h5ad, delegates inference to that CLI, then loads the resulting embeddings.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

import paths


def _checkpoint_dir() -> Path:
    value = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_CKPT", "").strip()
    model = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_MODEL", "tf_sapiens").strip()
    return Path(value).expanduser().resolve() if value else paths.pretrained_root() / "transcriptformer" / model


def _ensure_ensembl_column(adata: ad.AnnData, gene_col_name: str) -> ad.AnnData:
    if gene_col_name in adata.var.columns:
        return adata
    out = adata.copy()
    for candidate in ("ensembl_id", "ensemblid", "gene_id", "feature_id", "gene_ids"):
        if candidate in out.var.columns:
            out.var[gene_col_name] = out.var[candidate].astype(str).values
            return out
    if all(str(x).startswith(("ENS", "ENSG", "ENSMUSG")) for x in out.var_names[: min(50, out.n_vars)]):
        out.var[gene_col_name] = out.var_names.astype(str)
        return out
    raise ValueError(
        "TranscriptFormer requires Ensembl gene IDs. Add adata.var['ensembl_id'] "
        "or set LATENT_BENCH_TRANSCRIPTFORMER_GENE_COL to an existing var column."
    )


def _run_cli(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-20:] + (proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(f"TranscriptFormer inference failed with code {proc.returncode}:\n{tail}")


def encode(
    adata: ad.AnnData,
    *,
    device: str = "cuda",
    batch_size: int = 2,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return TranscriptFormer mean-pooled cell embeddings.

    TranscriptFormer expects raw counts. If ``input_is_log1p`` is true, this
    adapter relies on ``adata.raw`` being present and asks the official CLI to use
    ``AnnData.raw.X``.  Otherwise it uses ``adata.X`` directly.
    """
    del force_pert, show_progress
    ckpt = _checkpoint_dir()
    if not (ckpt / "config.json").is_file() or not (ckpt / "model_weights.pt").is_file():
        raise FileNotFoundError(
            f"TranscriptFormer checkpoint missing under {ckpt}. Download with: "
            f"transcriptformer download tf-sapiens --checkpoint-dir {ckpt.parent}"
        )

    gene_col = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_GENE_COL", "ensembl_id").strip()
    work_adata = _ensure_ensembl_column(adata, gene_col)
    use_raw = "auto"
    if input_is_log1p:
        if work_adata.raw is None:
            raise ValueError(
                "TranscriptFormer needs raw counts, but input_is_log1p=True and adata.raw is absent. "
                "Export this model from a raw-count h5ad or pass --no-input-is-log1p only when X is counts."
            )
        use_raw = "true"
    else:
        use_raw = "false"

    precision = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_PRECISION", "16-mixed")
    emb_type = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_EMB_TYPE", "cell")
    clip_counts = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_CLIP_COUNTS", "30")
    num_gpus = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_NUM_GPUS", "1")
    n_workers = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_N_DATA_WORKERS", "0")
    oom_loader = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_OOM_DATALOADER", "1") != "0"
    py = sys.executable

    third_party = paths.third_party_root() / "transcriptformer" / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{third_party}:{env.get('PYTHONPATH', '')}" if third_party.is_dir() else env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory(prefix="scfm_transcriptformer_") as td:
        tmp = Path(td)
        in_h5ad = tmp / "input.h5ad"
        out_dir = tmp / "out"
        out_name = "embeddings.h5ad"
        work_adata.write_h5ad(in_h5ad)
        cmd = [
            py,
            "-c",
            "import transcriptformer.cli as c; c.main()",
            "inference",
            "--checkpoint-path",
            str(ckpt),
            "--data-file",
            str(in_h5ad),
            "--output-path",
            str(out_dir),
            "--output-filename",
            out_name,
            "--batch-size",
            str(batch_size),
            "--gene-col-name",
            gene_col,
            "--precision",
            precision,
            "--use-raw",
            use_raw,
            "--emb-type",
            emb_type,
            "--num-gpus",
            num_gpus,
            "--device",
            "cuda" if device.startswith("cuda") else device,
            "--clip-counts",
            clip_counts,
            "--n-data-workers",
            n_workers,
        ]
        if oom_loader:
            cmd.append("--oom-dataloader")
        _run_cli(cmd, env)
        result = ad.read_h5ad(out_dir / out_name)

    if "embeddings" not in result.obsm:
        raise RuntimeError("TranscriptFormer output missing obsm['embeddings']")
    z = np.asarray(result.obsm["embeddings"], dtype=np.float32)
    if z.ndim != 2:
        raise RuntimeError(f"TranscriptFormer embeddings must be 2D, got shape {z.shape}")
    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "model_family": "TranscriptFormer",
        "official_repo": "https://github.com/czi-ai/transcriptformer",
        "checkpoint_path": str(ckpt),
        "checkpoint_model": ckpt.name,
        "pooling": "official cell mean-pooled embedding",
        "embedding_layer_index": "official_cli_default",
        "note": "README documents --embedding-layer-index, but current official argparse does not expose it.",
        "emb_type": emb_type,
        "gene_col_name": gene_col,
        "use_raw": use_raw,
        "precision": precision,
        "batch_size": int(batch_size),
        "num_gpus": int(num_gpus),
        "oom_dataloader": bool(oom_loader),
        "n_data_workers": int(n_workers),
        "input_is_log1p": bool(input_is_log1p),
        "third_party_src": str(third_party),
        "force_pert_effective": False,
        "pert_source": None,
    }
    return z, meta
