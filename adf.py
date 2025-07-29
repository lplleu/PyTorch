import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ----------- CONFIG ------------
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
NUM_CLASSES = 2
BATCH_SIZE = 50
NUM_EPOCHS = 10
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASS_LABELS = ['__background__', 'house']
# --------------------------------


# ----------- DATASET ------------
class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_file = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_file)
            ann_path = os.path.join(self.annotation_dir, os.path.splitext(img_file)[0] + ".xml")

            # Load image
            img = Image.open(img_path).convert("RGB")

            # Parse annotation
            boxes, labels = self.parse_voc_xml(ann_path)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }

            if self.transforms:
                img = self.transforms(img)

            return img, target

        except Exception as e:
            print(f"[Skipping] Error loading {img_file}: {e}")
            # Pick another index to ensure batch stays consistent
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

    def parse_voc_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in CLASS_LABELS:
                print(f"[Warning] Unknown label '{label}' found, skipping.")
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
# --------------------------------


# ----------- MODEL ------------
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model
# --------------------------------


# ----------- TRAINING UTILS ------------
def collate_fn(batch):
    return tuple(zip(*batch))

def compute_metrics_with_confusion_matrix(true_boxes, true_labels, pred_boxes, pred_labels, pred_scores, num_classes, iou_threshold, score_threshold):
    # Dummy placeholder implementation (returns dummy values)
    precision, recall, f1 = 0.0, 0.0, 0.0
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)  # +1 for No Detection class
    return precision, recall, f1, cm


def evaluate(model, data_loader, device, class_labels, num_classes, iou_threshold=0.5, score_threshold=0.5):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'].cpu())
                all_true_labels.append(target['labels'].cpu())
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())

            if i % 10 == 0:
                print(f"  Evaluation Iteration {i}/{len(data_loader)}")

    precision, recall, f1, cm = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        num_classes, iou_threshold, score_threshold
    )

    print(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels + ["No Detection"])
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1, cm


# ----------- MAIN TRAINING LOOP ------------
def train():
    dataset = CustomDataset(IMAGE_DIR, ANNOTATION_DIR, transforms=T.ToTensor())
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images, targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {running_loss / len(train_loader):.4f}")

        print("--- evaluation on validation set ---")
        evaluate(
            model,
            val_loader,
            device=DEVICE,
            class_labels=CLASS_LABELS,
            num_classes=NUM_CLASSES,
            iou_threshold=0.5,
            score_threshold=0.5
        )

    print("\n--- Final Evaluation on Validation Set ---")
    evaluate(model, val_loader, device=DEVICE, class_labels=CLASS_LABELS, num_classes=NUM_CLASSES)


if __name__ == "__main__":
    train()
