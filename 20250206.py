from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (if needed)
])

from torchvision.datasets import VOCDetection  # Example dataset

train_dataset = VOCDetection(root='path_to_data', year='2012', image_set='train', transform=transform)
val_dataset = VOCDetection(root='path_to_data', year='2012', image_set='val', transform=transform)
