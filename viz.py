import os
import torch
import torchvision.transforms as T
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def parse_voc_xml(path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in CLASS_LABELS:
            continue
        label_id = CLASS_LABELS.index(label)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_id)
    return boxes, labels


def visualize_detection_result(image_path, annotation_path, model_path, output_path,
                               iou_threshold=0.5, score_threshold=0.5):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_image = np.array(image)

    # Load ground truth annotations
    boxes_gt, labels_gt = parse_voc_xml(annotation_path)
    boxes_gt = torch.tensor(boxes_gt, dtype=torch.float32)
    labels_gt = torch.tensor(labels_gt, dtype=torch.int64)

    # Load model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    # Transform image
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)[0]

    pred_boxes = output['boxes'].cpu()
    pred_labels = output['labels'].cpu()
    pred_scores = output['scores'].cpu()

    # Filter predictions
    mask = pred_scores >= score_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # Match predictions to ground truth
    matched_gt = set()
    matches = []

    for i, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
        ious = ops.box_iou(pb.unsqueeze(0), boxes_gt)
        max_iou, max_idx = ious.max(1)
        if max_iou.item() >= iou_threshold and max_idx.item() not in matched_gt:
            matches.append(("TP", pb, pl, pred_scores[i]))
            matched_gt.add(max_idx.item())
        else:
            matches.append(("FP", pb, pl, pred_scores[i]))

    for i, (box, label) in enumerate(zip(boxes_gt, labels_gt)):
        if i not in matched_gt:
            matches.append(("FN", box, label, None))

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig_image)

    for match_type, box, label, score in matches:
        color = {"TP": "green", "FP": "red", "FN": "orange"}[match_type]
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label_text = f"{CLASS_LABELS[label]} ({match_type})"
        if score is not None:
            label_text += f" {score:.2f}"
        ax.text(x1, y1 - 5, label_text, color=color, fontsize=10, backgroundcolor="white")

    plt.title("Detection Result with TP / FP / FN")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Detection result saved at: {output_path}")

image_filename = "0001.jpg"
image_path = os.path.join(IMAGE_DIR, image_filename)
annotation_path = os.path.join(ANNOTATION_DIR, os.path.splitext(image_filename)[0] + ".xml")
model_path = os.path.join(CHECKPOINT_DIR, "best_model_map_epoch_7.pth")  # adjust as needed
output_path = os.path.join(CHECKPOINT_DIR, "annotated_result.png")

visualize_detection_result(
    image_path=image_path,
    annotation_path=annotation_path,
    model_path=model_path,
    output_path=output_path
)


