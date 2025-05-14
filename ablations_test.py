# Import packages
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Import local modules
from objects.video_dataset_loader import VideoDataset
from objects.models import AutoEncoder, UNet, Conv3dSkipUNet, FullConv2DMaskedSkipUNet, FullMaskedSkipUNet, OneMaskedSkipUNet, TwoMaskedSkipUNet, ThreeMaskedSkipUNet
from objects.video_writer import VideoWriter
from objects.scoring import Score
from objects.ped2_evaluator import Ped2Evaluator

# -----------------------------
# Parameters
# -----------------------------
data_dir = "path/to/your/testing/folder" # replace with actual path
ground_truth_file = "path/to/your/UCSDped2.m" # replace with actual path
resize = (256, 256)
batch_size = 1

transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor()
])

models = {}
models['AutoEncoder'] = (AutoEncoder, 1)
models['UNet'] = (UNet, 1)
models['Conv3dSkipUNet'] = (Conv3dSkipUNet, 3)
models['2D MaskedSkipUNet'] = (FullConv2DMaskedSkipUNet, 1)
models['E_D Masks MaskedSkipUNet'] = (FullMaskedSkipUNet, 3)
models['One Mask MaskedSkipUNet'] = (OneMaskedSkipUNet, 3)
models['Two Mask MaskedSkipUNet'] = (TwoMaskedSkipUNet, 3)
models['Three Mask MaskedSkipUNet'] = (ThreeMaskedSkipUNet, 3)

for model_name in models:

    model, temporal = models[model_name]
    video_output_dir = f"./reconstruction_videos/{model_name}"
    model_path = f"models/{model_name}.pt"


    # Dataset and DataLoader
    dataset = VideoDataset(directory=data_dir, temporal=temporal, resize=resize, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(latent_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize evaluator, loss, and video writer
    evaluator = Ped2Evaluator(gt_m_file=ground_truth_file)
    criterion = Score()
    os.makedirs(video_output_dir, exist_ok=True)
    video_writer = VideoWriter(video_output_dir)

    # Evaluation loop
    all_scores = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Testing {model_name}")
        for batch in progress_bar:
            sequences, paths = batch  
            sequences = sequences.to(device)
            outputs = model(sequences)
            recon_maps = criterion(sequences, outputs)
            
            # Use reconstruction error as anomaly score
            score = recon_maps[-1]
            all_scores.append(score)

            # Flatten all paths to one list (batch size = 1)
            flat_paths = list(paths[0]) if isinstance(paths[0], (list, tuple)) else [paths[0]]
            labels = evaluator.generate_labels(flat_paths)
            all_labels.append(max(labels)) 

            # Write visualization frame to video
            video_writer.update(flat_paths[-1], recon_maps)

    video_writer.release()


    # Compute AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n {model_name} Test AUC: {auc:.4f}\n")
