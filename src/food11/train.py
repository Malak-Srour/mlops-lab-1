# ------------------------------------------------------------
# train.py - Train a ResNet18 model on the Food-11 dataset
# and record everything (settings, results, model) in mlflow
#
# Part 1: read the settings and load the images
# Part 2: build the model
# Part 3: train the model and measure how good it is
# Part 4: record everything in mlflow
# ------------------------------------------------------------

# argparse lets us read settings typed in the terminal (like --lr 0.001)
import argparse

# Path helps us build folder paths that work on Windows, Mac and Linux
from pathlib import Path

# mlflow records our settings, results and model
import mlflow
# mlflow.pytorch knows how to save a PyTorch model
import mlflow.pytorch

# torch is the main deep learning library
import torch
# nn contains the building blocks of neural networks (layers)
from torch import nn

# DataLoader gives the images to the model in small groups (batches)
from torch.utils.data import DataLoader

# datasets: ready-made tools to read image folders
# models: ready-made famous models, like ResNet18
# transforms: tools to prepare images before the model sees them
from torchvision import datasets, models, transforms

# accuracy_score compares the true labels with the model's guesses
# and gives the percentage of correct answers
from sklearn.metrics import accuracy_score


# A small lookup table: the name we type -> the folder it means
DATA_DIRS = {
    "processed": Path("data/food11_processed"),       # full dataset
    "mini": Path("data/food11_processed_mini"),       # small dataset for testing
}

# Our dataset has 11 food categories
NUM_CLASSES = 11

# Where the mlflow server is, and which experiment to write in
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "food11"


# ============================================================
# Part 1: settings and data
# ============================================================

def parse_args():
    # Create a "reader" for the settings typed after the script name
    parser = argparse.ArgumentParser(description="Train ResNet18 on Food-11")

    # Which dataset to use: only "processed" or "mini" are allowed
    parser.add_argument("--dataset", choices=["processed", "mini"], default="mini")

    # How many times the model goes through all the training images
    parser.add_argument("--epochs", type=int, default=5)

    # Learning rate: how big each learning step is
    parser.add_argument("--lr", type=float, default=0.001)

    # How many images the model looks at in one go
    # Note: "--batch-size" becomes args.batch_size in Python (dash -> underscore)
    parser.add_argument("--batch-size", type=int, default=32)

    # Read what was typed and return all the settings together
    return parser.parse_args()


def get_dataloaders(dataset, batch_size):
    # Find the folder of the chosen dataset (mini or processed)
    data_dir = DATA_DIRS[dataset]

    # Steps applied to every image before it goes into the model
    transform = transforms.Compose([
        # Turn the picture into numbers between 0 and 1
        transforms.ToTensor(),
        # Adjust the colors the same way ResNet was originally trained,
        # so our images "look familiar" to the pretrained model
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder reads each subfolder name (Bread, Egg...) as the label
    # of the images inside it - that's why we organized them in Lab 1
    train_set = datasets.ImageFolder(data_dir / "training", transform=transform)    # to learn
    val_set = datasets.ImageFolder(data_dir / "validation", transform=transform)    # to check after each epoch
    test_set = datasets.ImageFolder(data_dir / "evaluation", transform=transform)   # final score at the end

    # Serve the images in batches
    # shuffle=True mixes the training images every epoch, so the order doesn't matter
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # No need to shuffle validation and test, we only measure on them
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Also return the list of class names (the 11 food categories)
    return train_loader, val_loader, test_loader, train_set.classes


# ============================================================
# Part 2: the model
# ============================================================

def build_model():
    # Load ResNet18 with weights already trained on ImageNet
    # (millions of photos of 1000 everyday objects)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # The last layer ("fc") gives 1000 outputs, one per ImageNet object.
    # We replace it with a new layer that gives 11 outputs, one per food category.
    # in_features = how many numbers come into this layer (512 for ResNet18)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


# ============================================================
# Part 3: training and evaluation
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer):
    # Put the model in "learning mode"
    model.train()
    total_loss = 0.0

    # Go through the training images, one batch at a time
    for images, labels in loader:
        # 1. Clear the corrections from the previous batch
        optimizer.zero_grad()
        # 2. The model guesses a category for each image
        outputs = model(images)
        # 3. Measure how wrong the guesses are
        loss = loss_fn(outputs, labels)
        # 4. Work out how each weight should change to reduce the mistake
        loss.backward()
        # 5. Apply those changes to the weights
        optimizer.step()

        # Add up the loss (multiplied by the batch size, to average correctly later)
        total_loss += loss.item() * images.size(0)

    # Return the average loss over all the training images
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn):
    # Put the model in "testing mode" (no learning)
    model.eval()
    total_loss = 0.0
    all_labels = []        # the true categories
    all_predictions = []   # the model's guesses

    # torch.no_grad() = we only measure, we don't learn, so it's faster
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # The guess = the category with the highest score out of the 11
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    average_loss = total_loss / len(loader.dataset)
    # Accuracy = fraction of images the model got right (between 0 and 1)
    accuracy = accuracy_score(all_labels, all_predictions)
    return average_loss, accuracy


# ============================================================
# Part 4: saving the model in mlflow
# ============================================================

def save_model_to_mlflow(model, test_loader):
    # Put the model in "testing mode" before saving it
    model.eval()

    # A small example of the model's input: 2 real images, as a numpy array.
    # mlflow uses it to record what the input and output look like
    # (this description is called the model "signature").
    example_images, _ = next(iter(test_loader))
    input_example = example_images[:2].numpy()

    try:
        # Save the model using the classic "pickle" format.
        # (The newer default format "pt2" is too strict for our simple setup,
        # that's what caused the errors before.)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=input_example,
            serialization_format="pickle",
        )
    except Exception as error:
        # Safety net: if mlflow has a problem with the input example,
        # save the model without it, so the run is never lost
        print("Could not save with input example, saving without it. Reason:", error)
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )


# ============================================================
# Main program: everything put together
# ============================================================

def main():
    # 1. Read the settings from the terminal
    args = parse_args()
    print("Settings:", args)

    # 2. Connect to the mlflow server and choose the experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # 3. Load the three sets of images
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.dataset, args.batch_size
    )
    print("Train images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    # 4. Build the model
    model = build_model()

    # 5. Choose how to measure mistakes and how to correct them
    # CrossEntropyLoss is the standard loss when choosing between categories
    loss_fn = nn.CrossEntropyLoss()
    # Adam is a popular optimizer: it adjusts the weights using the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. Start one mlflow run: everything inside this block is recorded together
    with mlflow.start_run() as run:
        print("mlflow run ID:", run.info.run_id)

        # Log the settings ONCE, at the start (params never change during the run)
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
        })

        # 7. Train for the chosen number of epochs
        for epoch in range(1, args.epochs + 1):
            # Learn from the training images
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            # Check on the validation images (never used for learning)
            val_loss, val_accuracy = evaluate(model, val_loader, loss_fn)

            print(f"Epoch {epoch}/{args.epochs} - "
                  f"train_loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - "
                  f"val_accuracy: {val_accuracy:.4f}")

            # Log the results of THIS epoch
            # step=epoch tells mlflow which epoch the value belongs to,
            # so it can draw a chart of how the value changes over time
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # 8. Final score on the test images (never seen during training)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Log the final test accuracy (only one value, at the end)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # 9. Save the trained model inside this run
        save_model_to_mlflow(model, test_loader)
        print("Model saved to mlflow.")


# Only run main() when this file is run directly
# (also avoids problems with DataLoader on Windows)
if __name__ == "__main__":
    main()