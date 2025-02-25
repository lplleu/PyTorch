import torch
import torchvision

# Initialize Faster R-CNN with pretrained weights
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze backbone if needed (optional)
for param in model.backbone.parameters():
    param.requires_grad = False  # Freeze backbone layers

# Make sure only the new layers require gradients
for param in model.roi_heads.parameters():
    param.requires_grad = True  # Allow gradients for ROI head

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
def train(model, data_loader, optimizer, num_epochs=5, model_path='/content/drive/MyDrive/all/results/frcnn_model_20250222.pth'):
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch+1}/{num_epochs} started...")

        for i, (images, targets) in enumerate(data_loader):
            # Move images and targets to the correct device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            optimizer.zero_grad()  # Clear gradients
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Backward pass
            losses.backward()
            optimizer.step()

            if i % 10 == 0:  # Provide feedback every 10 iterations
                print(f"  Iteration {i}/{len(data_loader)}, Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} finished, Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")
