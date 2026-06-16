"""NicheFormer expression-only adapter.

NicheFormer currently ships a Lightning model and tokenized-data workflow rather
than a stable h5ad inference CLI.  This adapter supports direct h5ad tokenization
only when the official checkpoint and model-mean gene order are present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

import paths


def _third_party_src() -> Path:
    return paths.third_party_root() / "nicheformer" / "src"


def _checkpoint_path() -> Path:
    value = os.environ.get("LATENT_BENCH_NICHEFORMER_CKPT", "").strip()
    return Path(value).expanduser().resolve() if value else paths.pretrained_root() / "nicheformer" / "nicheformer.ckpt"


def _mean_h5ad_path() -> Path:
    value = os.environ.get("LATENT_BENCH_NICHEFORMER_MEAN_H5AD", "").strip()
    default = paths.third_party_root() / "nicheformer" / "data" / "model_means" / "model.h5ad"
    return Path(value).expanduser().resolve() if value else default


def _aligned_counts_and_means(adata: ad.AnnData, mean_h5ad: Path, input_is_log1p: bool) -> tuple[ad.AnnData, np.ndarray]:
    if input_is_log1p:
        raise ValueError(
            "NicheFormer direct tokenization expects count-like X. Provide raw-count h5ad "
            "and pass --no-input-is-log1p; this adapter will not exponentiate log data implicitly."
        )
    if not mean_h5ad.is_file():
        raise FileNotFoundError(f"NicheFormer model mean h5ad missing: {mean_h5ad}")
    mean_adata = ad.read_h5ad(mean_h5ad)
    target_genes = mean_adata.var_names.astype(str)
    if mean_adata.X.shape[0] != 1:
        raise ValueError(f"Expected one-row NicheFormer mean h5ad, got {mean_adata.shape}")
    missing = [g for g in target_genes[:100] if g not in adata.var_names]
    if len(missing) == 100:
        raise ValueError(
            "NicheFormer mean genes do not match input var_names. Use Ensembl IDs as var_names "
            "or pre-align the AnnData to the NicheFormer model mean gene set."
        )
    aligned = adata[:, target_genes.intersection(adata.var_names)].copy()
    if aligned.n_vars < 1000:
        raise ValueError(f"Only {aligned.n_vars} NicheFormer genes overlap input; refusing to encode.")
    means = np.asarray(mean_adata[:, aligned.var_names].X).reshape(-1).astype(np.float32)
    return aligned, means


def encode(
    adata: ad.AnnData,
    *,
    device: str = "cuda",
    batch_size: int = 4,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean-pooled NicheFormer cell embeddings from the last layer."""
    del force_pert, show_progress
    ckpt = _checkpoint_path()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"NicheFormer checkpoint missing: {ckpt}. Official README points to Mendeley weights; "
            "place the pretrained .ckpt here or set LATENT_BENCH_NICHEFORMER_CKPT."
        )
    src = _third_party_src()
    if not src.is_dir():
        raise FileNotFoundError(f"NicheFormer source checkout missing: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from scipy.sparse import issparse
    from torch.utils.data import DataLoader, TensorDataset

    from nicheformer.data.dataset import sf_normalize, _sub_tokenize_data
    from nicheformer.models._nicheformer import Nicheformer

    aligned, means = _aligned_counts_and_means(adata, _mean_h5ad_path(), input_is_log1p)
    model = Nicheformer.load_from_checkpoint(str(ckpt), map_location="cpu")
    model.eval()
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(dev)

    aux_fields = ["specie", "assay", "modality"]
    aux_count = sum(bool(getattr(model.hparams, name, False)) for name in aux_fields)
    max_seq_len = int(model.hparams.context_length) - aux_count
    if max_seq_len < 128:
        raise ValueError(f"Invalid NicheFormer context length after aux tokens: {max_seq_len}")

    tokens_chunks: list[np.ndarray] = []
    chunk_size = int(os.environ.get("LATENT_BENCH_NICHEFORMER_TOKEN_CHUNK", "512"))
    for start in range(0, aligned.n_obs, chunk_size):
        chunk = aligned.X[start : start + chunk_size]
        x = chunk.toarray() if issparse(chunk) else np.asarray(chunk)
        x = sf_normalize(np.nan_to_num(x).astype(np.float32, copy=False))
        x = x / np.where(means == 0, 1.0, means).reshape(1, -1)
        tokens_chunks.append(_sub_tokenize_data(x, max_seq_len, 30).astype(np.int64))
    tokens = np.concatenate(tokens_chunks, axis=0)
    ds = TensorDataset(torch.from_numpy(tokens))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            batch: dict[str, torch.Tensor] = {"X": x_batch.to(dev)}
            for field in aux_fields:
                if bool(getattr(model.hparams, field, False)):
                    value = int(os.environ.get(f"LATENT_BENCH_NICHEFORMER_{field.upper()}_TOKEN", "0"))
                    batch[field] = torch.full((x_batch.shape[0],), value, dtype=torch.int64, device=dev)
            emb = model.get_embeddings(batch, layer=int(os.environ.get("LATENT_BENCH_NICHEFORMER_LAYER", "-1")))
            outputs.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
    z = np.concatenate(outputs, axis=0)
    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "model_family": "NicheFormer",
        "official_repo": "https://github.com/theislab/nicheformer",
        "checkpoint_path": str(ckpt),
        "mean_h5ad": str(_mean_h5ad_path()),
        "pooling": "official get_embeddings mean pooling",
        "layer": int(os.environ.get("LATENT_BENCH_NICHEFORMER_LAYER", "-1")),
        "n_overlap_genes": int(aligned.n_vars),
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "input_is_log1p": bool(input_is_log1p),
        "third_party_src": str(src),
        "force_pert_effective": False,
        "pert_source": None,
    }
    return z, meta
