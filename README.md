# Unified Streamlit Pipeline

Single Streamlit application that combines three components trained on the Qwen3-VL-2B-Instruct backbone:

1. **Summarization** — Qwen3-VL + Q-Former LoRA checkpoint (`checkpoint-400`).
2. **Multi-class classification** — `EnhancedTemporalAdapterModel` that consumes the generated summary embedding plus frame embeddings.
3. **Binary classification** — same `SimpleAdapter` + `LoRA` ensemble used by `run_streamlit_v3.py`.

## Files

- `pipeline_base.py` — extracted reusable `EnhancedTemporalAdapterModel` and embedding helpers.
- `pipeline.py` — UI-agnostic core that loads all models and runs the full pipeline.
- `streamlit_app_unified.py` — Streamlit user interface.
- `run_streamlit_unified.py` — deterministic launcher.
- `binary_ensemble.py` — binary ensemble (copied from the v3 Streamlit project).
- `requirements.txt` — Python dependencies.

## Quick Start

```bash
# Set paths to your model weights
export QWEN_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
export SUMMARIZATION_CHECKPOINT="/path/to/ucf_qwen_v9_qformer/checkpoint-400"
export MULTICLASS_MODEL_DIR="/path/to/models_ucf_v2/hpo_best_models"
export BINARY_MODEL_DIR="/path/to/binary_model"
export VIDEO_DIR="/path/to/videos"

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
python run_streamlit_unified.py
```

## Model Weights

Model weights are not committed to this repository (they are excluded by `.gitignore`).
Upload the model files to the configured paths or set the environment variables above
before launching the app.

## Notes

- `generate_summary` uses `do_sample=False` for deterministic output.
- `pipeline.py` sets `torch.backends.cudnn.deterministic = True` and seeds all random sources.
- Binary classification uses the same `simple_adapter.pt` + `lora_model.pt` ensemble as the deterministic v3 app.
