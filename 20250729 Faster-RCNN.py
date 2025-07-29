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
CLASS_LABELS = ['__background__', 'mokolwane']
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



def evaluate(model, dataloader):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            # Move to CPU for torchmetrics
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(outputs, targets)

    results = metric.compute()
    print("\n--- Mean Average Precision (mAP) Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

# --------------------------------


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
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    print("\n--- Evaluation on Validation Set ---")
    evaluate(model, val_loader)


if __name__ == "__main__":
    train()



