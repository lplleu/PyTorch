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


import optuna
from ultralytics import YOLO
import os

# Define the objective function
def objective(trial):
    # Hyperparameter search space
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)  # Learning rate
    momentum = trial.suggest_float("momentum", 0.6, 0.98)  # Momentum
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)  # L2 regularization
    epochs = trial.suggest_int("epochs", 10, 50)  # Number of epochs
    img_size = trial.suggest_int("img_size", 320, 640, step=64)  # Image size

    # Create the model and train it with suggested hyperparameters
    model = YOLO("yolov8n.yaml")  # Use the desired model configuration
    results = model.train(
        data="data.yaml",  # Path to the dataset configuration
        epochs=epochs,
        imgsz=img_size,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        project="optuna_results",  # Save each trial's results here
        name=f"trial_{trial.number}",  # Unique name for each trial
        verbose=False
    )

    # Return the evaluation metric (e.g., mAP@0.5) for Optuna to optimize
    return results.metrics.mAP50  # Replace with mAP@50-95 if needed

# Define the Optuna study
study = optuna.create_study(direction="maximize")  # Maximize mAP
study.optimize(objective, n_trials=20)  # Run 20 trials

# Print the best hyperparameters
print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")
