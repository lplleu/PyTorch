import optuna
from ultralytics import YOLO

def objective(trial):
    # Suggest hyperparameters
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)  # Initial learning rate
    momentum = trial.suggest_float("momentum", 0.6, 0.98)  # Momentum
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01)  # Weight decay
    epochs = trial.suggest_int("epochs", 10, 50)  # Number of epochs

    # Train YOLO model with suggested hyperparameters
    model = YOLO('yolov8n.yaml')  # Define your YOLO model here
    results = model.train(
        data="data.yaml",  # Path to your dataset configuration file
        epochs=epochs,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        project="optuna_results",  # Save results in a specific folder
        name=f"trial_{trial.number}",  # Trial-specific folder
        verbose=False
    )

    # Return validation performance (e.g., mAP@0.5)
    return results.metrics.mAP50  # Adjust based on the YOLO framework you're using

# Run Optuna study
study = optuna.create_study(direction="maximize")  # Maximize mAP
study.optimize(objective, n_trials=20)

# Print best trial results
print("Best hyperparameters:", study.best_params)
