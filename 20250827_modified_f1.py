from sklearn.metrics import confusion_matrix, classification_report

def compute_metrics_with_confusion_matrix(true_boxes_list, true_labels_list,
                                          pred_boxes_list, pred_labels_list, pred_scores_list,
                                          num_classes, iou_threshold=0.5, score_threshold=0.5,
                                          label_map=None):
    y_true, y_pred = [], []

    # Match predictions to ground truth
    for true_boxes, true_labels, pred_boxes, pred_labels, pred_scores in zip(
        true_boxes_list, true_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list):

        # Filter predictions by score threshold
        pred_mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[pred_mask]
        pred_labels = pred_labels[pred_mask]

        matched_gt = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            ious = ops.box_iou(pb.unsqueeze(0), true_boxes)
            max_iou, max_idx = ious.max(1)
            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                matched_gt.add(max_idx.item())
                y_true.append(true_labels[max_idx.item()].item())
                y_pred.append(pl.item())
            else:
                y_true.append(num_classes)  # "No Detection"
                y_pred.append(pl.item())

        # Add unmatched ground-truth boxes as false negatives
        for i, tl in enumerate(true_labels):
            if i not in matched_gt:
                y_true.append(tl.item())
                y_pred.append(num_classes)  # Predicted nothing

    # Classes excluding background, but include "No Detection"
    valid_classes = list(range(1, num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes + [num_classes])

    # Classification report for precision/recall/F1 per class
    report = classification_report(y_true, y_pred, labels=valid_classes, output_dict=True, zero_division=0)

    # Overall weighted metrics
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    # Per-class metrics (with class names if label_map provided)
    per_class_metrics = {}
    for cls_id in valid_classes:
        cls_name = label_map[cls_id] if label_map else str(cls_id)
        if str(cls_id) in report:
            per_class_metrics[cls_name] = {
                "precision": report[str(cls_id)]["precision"],
                "recall": report[str(cls_id)]["recall"],
                "f1": report[str(cls_id)]["f1-score"]
            }
        else:
            per_class_metrics[cls_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return precision, recall, f1, cm, per_class_metrics
