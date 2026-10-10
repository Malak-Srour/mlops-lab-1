# ------------------------------------------------------------
# copy_model_to_compose.py  (Lab 4)
# The mlflow container starts empty, so this copies a model trained
# in Lab 2 into it. No retraining: the model is loaded from its files
# on the laptop (mlruns/<experiment>/models/m-.../artifacts) and logged
# again as a new run in the mlflow container.
# Registering it and setting the alias is then done in the mlflow UI.
#
# Usage:  uv run python src/food11/copy_model_to_compose.py <model-id-start>
# Example: uv run python src/food11/copy_model_to_compose.py m-9e082f2c
# ------------------------------------------------------------
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np

# 1. Find the model files on the laptop
model_id = sys.argv[1]
matches = list(Path("mlruns").glob(f"*/models/{model_id}*/artifacts"))
if len(matches) != 1:
    sys.exit(f"Expected 1 model folder for '{model_id}', found {len(matches)}: {matches}")
model_dir = matches[0]
print(f"Model files found: {model_dir}")

# 2. Load the model from these local files (no retraining)
model = mlflow.pytorch.load_model(str(model_dir))

# 3. Log it as a new run in the mlflow container (port 5000 published by compose)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
with mlflow.start_run(run_name=f"copy-of-{model_dir.parent.name[:10]}") as run:
    mlflow.set_tag("copied_from_model", model_dir.parent.name)
    example = np.zeros((2, 3, 128, 128), dtype=np.float32)  # shape of the input
    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example,
        serialization_format="pickle",
        pip_requirements=str(model_dir / "requirements.txt"),
    )

print(f"Logged as run '{run.info.run_name}' ({run.info.run_id}) in http://127.0.0.1:5000")