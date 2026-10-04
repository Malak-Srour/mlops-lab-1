# ------------------------------------------------------------
# publish_for_serving.py
# Copies the current champion model into an experiment whose files
# are served by the mlflow server over HTTP (artifact proxy mode),
# registers it as a new version of food11, and moves "champion" to it.
# No retraining: we load the existing model and log it again.
# ------------------------------------------------------------
import mlflow
import mlflow.pytorch
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

# 1. Find the current champion (version 1) and load it.
#    This works here because its files are on this laptop.
old = client.get_model_version_by_alias("food11", "champion")
print(f"Current champion: food11 version {old.version} (run {old.run_id})")
model = mlflow.pytorch.load_model("models:/food11@champion")

# 2. Use a NEW experiment. It is created while the server is in proxy mode,
#    so its files get an "mlflow-artifacts:/" address, served over HTTP.
mlflow.set_experiment("food11-serving")
experiment = client.get_experiment_by_name("food11-serving")
print("New experiment artifact location:", experiment.artifact_location)

with mlflow.start_run(run_name="champion-for-serving"):
    # Keep track of where this model comes from
    mlflow.set_tags({
        "copied_from_model_version": old.version,
        "source_run_id": old.run_id,
    })
    # Copy the params and final metrics of the original training run
    if old.run_id:
        source_run = client.get_run(old.run_id)
        mlflow.log_params(source_run.data.params)
        mlflow.log_metrics(source_run.data.metrics)

    # 3. Log the same model again and register it as a new version of food11
    example = np.zeros((2, 3, 128, 128), dtype=np.float32)  # shape of the input
    info = mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=example,
        serialization_format="pickle",
        registered_model_name="food11",
    )

new_version = info.registered_model_version
print(f"Registered food11 version {new_version}")

# 4. Move the champion alias to the new version
client.set_registered_model_alias("food11", "champion", new_version)
champion = client.get_model_version_by_alias("food11", "champion")
print(f"champion -> food11 version {champion.version}")