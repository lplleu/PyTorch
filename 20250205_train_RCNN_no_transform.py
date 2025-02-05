import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.ops import box_iou
import os
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, target_dir):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def get_label_id(self, label):
        label_map = {
            'mokolwane': 0,
            'mopororo': 1,
            'motswere': 2,
        }
        return label_map.get(label, 0)  # 0 for unknown labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name).convert("RGB")

        target_name = os.path.join(self.target_dir, self.image_names[idx].replace('.jpg', '.xml'))
        target = self.load_annotation(target_name)

        # No transformation is applied as the dataset is already preprocessed.
        image = torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float()  # Normalize manually if needed.

        return image, target

    def load_annotation(self, target_file):
        tree = ET.parse(target_file)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Validate bounding box dimensions
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                label = obj.find('name').text
                labels.append(self.get_label_id(label))
            else:
                print(f"Invalid bounding box skipped: {[xmin, ymin, xmax, ymax]} in {target_file}")

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([0]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64)
        }

        return target


# Load datasets
train_dataset = CustomDataset(
    image_dir="datasets/all/train/images_2",  # point to the transformed image directory
    target_dir="datasets/all/train/annotations_1"  # point to the transformed annotations directory
)

val_dataset = CustomDataset(
    image_dir="datasets/all/val/images_2",  # point to the transformed image directory
    target_dir="datasets/all/val/annotations_1"  # point to the transformed annotations directory
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained SSD model
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()  # Set to evaluation mode initially

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001)

# Training loop
def train(model, data_loader, optimizer, num_epochs=5, model_path='all/frcnn_model_20250203.pth'):
    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        total_loss = 0
        print(f"Epoch {epoch+1}/{num_epochs} started...")

        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]  # .to(device) on the tensor
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:  # Provide feedback every 10 iterations
                print(f"  Iteration {i}/{len(data_loader)}, Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} finished, Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

# Confusion matrix and evaluation
def compute_metrics_with_confusion_matrix(true_boxes, true_labels, pred_boxes, pred_labels, pred_scores, num_classes, iou_threshold=0.5, score_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    all_true_labels, all_pred_labels = [], []

    for true_b, true_l, pred_b, pred_l, pred_s in zip(true_boxes, true_labels, pred_boxes, pred_labels, pred_scores):
        pred_b = pred_b[pred_s >= score_threshold]
        pred_l = pred_l[pred_s >= score_threshold]

        if len(pred_b) == 0:
            fn += len(true_l)
            all_true_labels.extend(true_l.tolist())
            all_pred_labels.extend([num_classes] * len(true_l))
            continue

        if len(true_b) == 0:
            fp += len(pred_l)
            all_pred_labels.extend(pred_l.tolist())
            all_true_labels.extend([num_classes] * len(pred_l))
            continue

        ious = box_iou(true_b, pred_b)
        matched = ious >= iou_threshold

        tp += matched.sum().item()
        fp += (len(pred_b) - matched.sum().item())
        fn += (len(true_b) - matched.sum().item())

        for t_idx, true_label in enumerate(true_l):
            if matched[t_idx].any():
                pred_idx = matched[t_idx].nonzero(as_tuple=True)[0][0]
                all_true_labels.append(true_label.item())
                all_pred_labels.append(pred_l[pred_idx].item())
            else:
                all_true_labels.append(true_label.item())
                all_pred_labels.append(num_classes)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(num_classes)) + [num_classes])
    return precision, recall, f1, cm

def evaluate_with_confusion_matrix(model, data_loader, num_classes, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_true_boxes, all_true_labels = [], []
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]  # .to(device) on the tensor
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for target, output in zip(targets, outputs):
                all_true_boxes.append(target['boxes'])
                all_true_labels.append(target['labels'])
                all_pred_boxes.append(output['boxes'].detach())
                all_pred_labels.append(output['labels'].detach())
                all_pred_scores.append(output['scores'].detach())

            if i % 10 == 0:  # Provide feedback every 10 iterations
                print(f"  Evaluation Iteration {i}/{len(data_loader)}")

    precision, recall, f1, cm = compute_metrics_with_confusion_matrix(
        all_true_boxes, all_true_labels, all_pred_boxes, all_pred_labels, all_pred_scores,
        num_classes, iou_threshold, score_threshold
    )

    print(f"Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    class_names = [f"Class {i}" for i in range(num_classes)] + ["No Detection"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1

# Training and evaluation
train(model, train_loader, optimizer, num_epochs=250)
evaluate_with_confusion_matrix(model, val_loader, num_classes=3)
