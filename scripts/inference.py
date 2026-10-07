"""Portable final prediction entry point; apply the policy after assembling the full batch."""
from pathlib import Path
from day5_final import predict_final


def predict(frame, artifact_dir=None):
    artifact_dir=Path(artifact_dir) if artifact_dir else Path(__file__).resolve().parents[1]/'artifacts/final_model'
    return predict_final(frame,artifact_dir)
