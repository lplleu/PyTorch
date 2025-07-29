#to add before train

import matplotlib.pyplot as plt

history = {
    "epoch": [],
    "loss": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "map_50": [],
    "map": []
}

best_model_data = None  # will store model + val_loader for best epoch

# to replace evaluate() block
print("\n--- Evaluating ---")
precision, recall, f1, _ = compute_metrics_with_confusion_matrix(
    all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
    NUM_CLASSES, iou_threshold=0.5, score_threshold=0.5
)

map_50, map_all = evaluate(
    model,
    val_loader,
    device=DEVICE,
    class_labels=CLASS_LABELS,
    num_classes=NUM_CLASSES,
    score_threshold=0.5
)

history["epoch"].append(epoch + 1)
history["loss"].append(avg_loss)
history["precision"].append(precision)
history["recall"].append(recall)
history["f1"].append(f1)
history["map_50"].append(map_50)
history["map"].append(map_all)

if map_50 > best_f1:
    best_f1 = map_50
    model_path = os.path.join(CHECKPOINT_DIR, f"best_model_map_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    best_model_data = {
        "model_path": model_path,
        "epoch": epoch + 1
    }
    print(f"✔️  Best model saved at epoch {epoch+1} with mAP@0.5: {map_50:.4f}")

# end of training
print("\n✅ Training complete. Plotting results...")

def plot_metric_curve(epochs, values, title, ylabel, filename):
    plt.figure()
    plt.plot(epochs, values, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, filename))
    plt.close()

# Plot all curves
plot_metric_curve(history["epoch"], history["loss"], "Training Loss", "Loss", "loss_curve.png")
plot_metric_curve(history["epoch"], history["precision"], "Precision", "Precision", "precision_curve.png")
plot_metric_curve(history["epoch"], history["recall"], "Recall", "Recall", "recall_curve.png")
plot_metric_curve(history["epoch"], history["f1"], "F1 Score", "F1", "f1_curve.png")
plot_metric_curve(history["epoch"], history["map_50"], "mAP@0.5", "mAP@0.5", "map50_curve.png")
plot_metric_curve(history["epoch"], history["map"], "mAP@[.5:.95]", "mAP", "map_curve.png")


#reload best model
if best_model_data:
    print(f"\n📌 Reloading best model from epoch {best_model_data['epoch']} for confusion matrix...")
    model.load_state_dict(torch.load(best_model_data["model_path"]))
    model.to(DEVICE)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'].cpu())
                all_true_labels.append(target['labels'].cpu())
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())

    precision, recall, f1, cm = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        NUM_CLASSES, iou_threshold=0.5, score_threshold=0.5
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_LABELS[1:] + ["No Detection"]
    )
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Best Epoch {best_model_data['epoch']}")
    plt.savefig(os.path.join(CHECKPOINT_DIR, "confusion_matrix.png"))
    plt.show()
