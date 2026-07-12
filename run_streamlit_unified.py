#!/usr/bin/env python
"""Deterministic launcher for the unified Streamlit pipeline.

Sets environment variables that make the output deterministic and then
runs `streamlit run streamlit_app_unified.py`.

Override model paths with environment variables before running:
  SET BINARY_MODEL_DIR=C:\path\to\binary_model
  SET SUMMARIZATION_CHECKPOINT=C:\path\to\checkpoint-400
  SET MULTICLASS_MODEL_DIR=C:\path\to\hpo_best_models
  python run_streamlit_unified.py
"""
import os
import sys
import subprocess

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Ensure the script directory is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app_unified.py"]
if __name__ == "__main__":
    subprocess.run(cmd, cwd=script_dir)
