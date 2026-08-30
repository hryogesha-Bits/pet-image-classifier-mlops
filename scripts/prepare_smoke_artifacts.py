from pathlib import Path

from PIL import Image
import torch

from src.model import build_model, save_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "data" / "ci"
MODEL_PATH = MODEL_DIR / "model.pt"
SAMPLE_PATH = SAMPLE_DIR / "sample.jpg"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # A deterministic, randomly initialized baseline model is sufficient for
    # CI/CD inference smoke tests. Production evaluation uses the trained model.
    torch.manual_seed(42)
    model = build_model(model_name="baseline_cnn")
    save_model(model, MODEL_PATH)

    # Keep a tiny, repository-generated image out of the real dataset.
    Image.new("RGB", (224, 224), color=(120, 80, 40)).save(SAMPLE_PATH, format="JPEG")

    print(f"Created smoke-test model: {MODEL_PATH}")
    print(f"Created smoke-test image: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
