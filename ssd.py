from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

def get_model(num_classes):
    model = ssd300_vgg16(pretrained=True)
    
    # Replacing classification head
    in_channels = [512, 1024, 512, 256, 256, 256]  # fixed for SSD300 VGG16
    num_anchors = model.head.classification_head.num_anchors  # [4, 6, 6, 6, 4, 4]
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    
    return model

####alternative
from torchcv.models.detection import ssd512
import torch

def get_model(num_classes):
    # pretrained on COCO
    model = ssd512(pretrained=True)
    # replacing classifier for dataset
    model.num_classes = num_classes
    return model

# forward
images = torch.randn(1, 3, 512, 512)
outputs = model(images)   # list of detections



#####
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])
