# Adapter artifacts

The package contains format-2 task metadata, the trained adapter or full model, and the saved tokenizer. Its integrity manifest records every file size and SHA-256 digest. The base repository and pinned revision remain explicit in the package metadata.

```bash
python koscope.py artifact package --checkpoint outputs/dpo/last.pt --destination exports/dpo
python koscope.py artifact verify --directory exports/dpo
python koscope.py artifact merge --checkpoint exports/dpo/last.pt --destination exports/merged --device cuda
```

Merged export loads the base on CUDA, applies the learned adapter and writes a full safetensors model. Base weights download automatically unless a local snapshot and offline mode are supplied. `artifact inspect` reads tensor headers and tokenizer metadata. `artifact compare` compares configuration and curriculum state between tasks.
