#from torchmetrics.detection.mean_ap import MeanAveragePrecision
#from sklearn.metrics import ConfusionMatrixDisplay
#import torch

def evaluate(model, data_loader, device, class_labels, num_classes,
             iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5])
    map_metric.to(device)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'].cpu())
                all_true_labels.append(target['labels'].cpu())
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())
                map_metric.update([output], [target])

    # Compute metrics
    precision, recall, f1, cm, per_class_metrics = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        num_classes, iou_threshold, score_threshold, label_map={i: name for i, name in enumerate(class_labels)}
    )

    map_result = map_metric.compute()
    map_50 = map_result["map_50"].item()
    map_all = map_result["map"].item()

    # Print overall metrics
    print(f"\nValidation Results:")
    print(f"Overall Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"mAP@0.5: {map_50:.4f}, mAP: {map_all:.4f}\n")

    # Print per-class metrics
    print("Per-Class Metrics:")
    for cls_name, metrics in per_class_metrics.items():
        print(f"  {cls_name:<25} "
              f"P: {metrics['precision']:.4f}, "
              f"R: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}")

    # Prepare labels for confusion matrix
    disp_labels = class_labels[1:] + ["No Detection"]
    assert cm.shape[0] == len(disp_labels), "Confusion matrix label mismatch"

    # Optional: plot confusion matrix (currently commented)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1, cm, map_50, map_all, per_class_metrics
