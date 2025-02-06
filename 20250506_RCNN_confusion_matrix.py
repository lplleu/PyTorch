from torchvision.ops import box_iou
import torch

def compute_metrics_with_confusion_matrix(
    all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
    num_classes, iou_threshold=0.5, score_threshold=0.5
):
    # Initialize confusion matrix (num_classes + 1 to account for "No Detection")
    cm = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int32)

    # Loop over all samples
    for true_boxes, true_labels, pred_boxes, pred_labels, pred_scores in zip(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores
    ):
        detected = set()  # Keep track of detected true boxes
        for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
            if ps < score_threshold:  # Skip predictions below score threshold
                continue

            ious = box_iou(pb.unsqueeze(0), true_boxes).squeeze(0)  # IoU of prediction with all true boxes
            best_iou, best_idx = ious.max(0)  # Find the best IoU and corresponding true box

            if best_iou >= iou_threshold and best_idx.item() not in detected:
                cm[true_labels[best_idx].item(), pl.item()] += 1  # True positive
                detected.add(best_idx.item())
            else:
                cm[-1, pl.item()] += 1  # False positive (no match with true boxes)

        for j in range(len(true_boxes)):
            if j not in detected:
                cm[true_labels[j].item(), -1] += 1  # False negative (true box not detected)

    # Calculate precision, recall, and F1 score
    true_positive = cm.diag()[:-1]  # Exclude "No Detection" row/column
    precision = true_positive.sum() / (cm.sum(1)[:-1].sum() + 1e-6)  # Avoid division by zero
    recall = true_positive.sum() / (cm.sum(0)[:-1].sum() + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item(), cm
