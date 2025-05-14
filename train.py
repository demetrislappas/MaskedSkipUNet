# Import packages
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import local
from objects.video_dataset_loader import VideoDataset
from objects.models import MaskedSkipUNet
from objects.loss_functions import NormLoss

# Parameters
data_dir = "path/to/your/training/folder"  # replace with actual path
resize = (256, 256)
temporal = 3
batch_size = 16
epochs = 10
learning_rate = 1e-2

# Define transform for resizing and normalization
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = VideoDataset(directory=data_dir, temporal=temporal, resize=resize, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedSkipUNet(latent_dim=512, temporal=temporal).to(device)
criterion = NormLoss(temporal)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Ensure models directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in progress_bar:
        sequences, _ = batch
        sequences = sequences.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(sequences,outputs)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    # Save model at the end of each epoch
    torch.save(model.state_dict(), os.path.join(model_dir, f"maskedskipunet_epoch_{epoch+1}.pt"))
