def evaluate_with_confusion_matrix(model, data_loader, num_classes, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'])
                all_true_labels.append(target['labels'])
                all_pred_boxes.append(output['boxes'].detach())
                all_pred_labels.append(output['labels'].detach())
                all_pred_scores.append(output['scores'].detach())

            if i % 10 == 0:
                print(f"  Evaluation Iteration {i}/{len(data_loader)}")

    precision, recall, f1, cm = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        num_classes, iou_threshold, score_threshold
    )

    print(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Convert to NumPy array for ConfusionMatrixDisplay
    cm_numpy = cm.cpu().numpy()
    class_names = [f"Class {i}" for i in range(num_classes)] + ["No Detection"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_numpy, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()
