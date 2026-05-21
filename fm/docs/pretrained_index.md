# Pretrained Asset Index

Default root: `SCFM_PRETRAINED_ROOT`, falling back to
`<delivery_root>/scFM_pretrained`. Legacy `COUPLEDFM_PRETRAINED_ROOT` remains an
explicit override.

| Model | Required path under `SCFM_PRETRAINED_ROOT` |
| --- | --- |
| Geneformer V2-316M | `geneformer/Geneformer-V2-316M/` |
| UCE | `uce/model_files/33layer_model.torch` and companion token/protein files |
| State SE-600M | `state/SE-600M/` with `.ckpt` or `.safetensors`; optional `protein_embeddings.pt` |
| arc-stack | `stack/bc_large.ckpt`, `stack/basecount_1000per_15000max.pkl` |
| scLDM | `scdlm/vae_census/{70M.ckpt,70M.yaml,concatenated_unique_genes.parquet}` |
| xVERSE | `xVerse/xVERSE_384.pth` |
| scGPT | `scgpt/{best_model.pt,vocab.json,args.json}` |
| CellNavi | `cellnavi/data/pretrain/pretrain_weights.pth`, `cellnavi/data/gene_name.txt`, `cellnavi/data/Nichenet/{node2idx.json,graph.pkl}` |
| scFoundation | `scFoundation/models.ckpt` |
| NicheNet standalone | `nichenet/{node2idx.json,idx2node.json,graph.pkl,graph.pt}` |

Third-party source code is separate and should be placed under
`SCFM_THIRD_PARTY_ROOT` (`<delivery_root>/scFM_third_party` by default).

Validate with:

```bash
PYTHONPATH=/path/to/scFM/fm python -m tools.validate_resources --skip-import-test
```
