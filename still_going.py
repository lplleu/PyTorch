import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
import torchvision.ops as ops

# ----------- CONFIG ------------
IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_EPOCHS = 40
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASS_LABELS = ['__background__', 'house']
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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

            img = Image.open(img_path).convert("RGB")
            boxes, labels = self.parse_voc_xml(ann_path)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

            if self.transforms:
                img = self.transforms(img)

            return img, target

        except Exception as e:
            print(f"[Skipping] Error loading {img_file}: {e}")
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

def compute_metrics_with_confusion_matrix(true_boxes_list, true_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list,
                                          num_classes, iou_threshold=0.5, score_threshold=0.5):
    y_true, y_pred = [], []

    for true_boxes, true_labels, pred_boxes, pred_labels, pred_scores in zip(
        true_boxes_list, true_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list):

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
                y_true.append(num_classes)
                y_pred.append(pl.item())

        for i, tl in enumerate(true_labels):
            if i not in matched_gt:
                y_true.append(tl.item())
                y_pred.append(num_classes)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)) + [num_classes])
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    return precision, recall, f1, cm


def evaluate(model, data_loader, device, class_labels, num_classes, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'].cpu())
                all_true_labels.append(target['labels'].cpu())
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())

    precision, recall, f1, cm = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        num_classes, iou_threshold, score_threshold
    )

    print(f"\nValidation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels[1:] + ["No Detection"])
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1, cm
# --------------------------------


# ----------- MAIN TRAINING LOOP ------------
def train():
    writer = SummaryWriter()

    train_transforms = T.Compose([
        T.Resize((512, 512)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    dataset = CustomDataset(IMAGE_DIR, ANNOTATION_DIR, transforms=train_transforms)
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

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        lr_scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Training Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        print("\n--- Evaluating ---")
        precision, recall, f1, _ = evaluate(
            model,
            val_loader,
            device=DEVICE,
            class_labels=CLASS_LABELS,
            num_classes=NUM_CLASSES,
            iou_threshold=0.5,
            score_threshold=0.5
        )

        writer.add_scalar("Metrics/Precision", precision, epoch + 1)
        writer.add_scalar("Metrics/Recall", recall, epoch + 1)
        writer.add_scalar("Metrics/F1", f1, epoch + 1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.pth"))
            print(f"✔️  Best model saved at epoch {epoch+1} with F1: {f1:.4f}")

    writer.close()


if __name__ == "__main__":
    train()
