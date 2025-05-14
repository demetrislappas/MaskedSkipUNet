# MaskedSkipUNet for Video Anomaly Detection

This repository provides an implementation of **MaskedSkipUNet**, a novel architecture for video anomaly detection that incorporates **MaskedConv3D layers** within skip connections of a UNet. The method is detailed in the research paper:

📄 **[Masked Convolutions within Skip Connections for Video Anomaly Detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4768663)**
by *Demetris Lappas, Dimitrios Makris, and Vasileios Argyriou*

## 📌 Overview

MaskedSkipUNet addresses a key limitation in traditional UNets: the tendency to reconstruct anomalous regions as faithfully as normal ones due to skip connections. By inserting MaskedConv3D layers into these connections, the model infers missing information from surrounding context, improving its ability to isolate and detect anomalies.

This repository provides:

* A PyTorch implementation of the architecture
* Scripts for training and testing
* Evaluation tools for Ped2 dataset using UCSDped2.m ground truth
* A reconstruction video visualizer

## 💪 Results

On the **UCSD Ped2** dataset, MaskedSkipUNet achieves **98.4% AUC**, matching or exceeding state-of-the-art methods. See the full paper for results on Avenue and ShanghaiTech datasets.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/demetrislappas/MaskedSkipUNet.git
cd MaskedSkipUNet
```

### 2. Create a Python Environment (Optional)

```bash
conda create -n maskedskipunet python=3.10
conda activate maskedskipunet
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA

Visit the [official PyTorch installation page](https://pytorch.org/get-started/locally/) and install the appropriate version for your system. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Training

To begin training:

```bash
python train.py
```

The script will:

* Load video sequences from the training directory specified in `train.py`
* Resize frames to (256, 256) and convert them to tensors
* Train `MaskedSkipUNet` for 10 epochs using the `NormLoss`
* Save model checkpoints at the end of each epoch to the `models/` directory

Make sure to update the `data_dir` variable in `train.py` to point to the UCSD Ped2 training set:

```python
data_dir = "/path/to/UCSDped2/Train"
```

You can also modify:

* `batch_size` (default = 16)
* `learning_rate` (default = 1e-2)
* `epochs` (default = 10)

All parameters are adjustable inside the `train.py` script.
. Adjust parameters such as batch size, learning rate, and number of epochs in the script as needed.

### Testing

To evaluate the model:

```bash
python test.py
```

The script will:

* Load the pretrained model from `models/maskedskipunet_epoch_10.pt`
* Evaluate on the UCSD Ped2 test set (set `data_dir` accordingly)
* Use reconstruction loss to compute anomaly scores per temporal clip
* Compare predictions with ground truth from `UCSDped2.m`
* Compute and display the final AUC
* Write visual comparison videos to the `reconstruction_videos/` directory

Make sure to update the following paths in `test.py` before running:

```python
data_dir = "/path/to/UCSDped2/Test"
ground_truth_file = "/path/to/UCSDped2.m"
```

All other hyperparameters (e.g., `temporal`, `resize`, and `batch_size`) can also be adjusted in the script.

* Load the pretrained model
* Evaluate on the UCSD Ped2 test set
* Generate reconstruction-based anomaly scores
* Compare with ground truth from `UCSDped2.m`
* Compute and print AUC
* Save reconstruction comparison videos under `reconstruction_videos/`

---

## 📂 File Structure

```text
.
├── README.md               # Project documentation
├── LICENSE                 # License terms
├── requirements.txt        # Package requirements
├── train.py                # Training entry point
├── test.py                 # Testing entry point
├── ablations_train.py      # Ablation training experiments
├── ablations_test.py       # Ablation testing experiments
└── objects/                # Core model and utility modules
    ├── video_writer.py
    ├── ped2_evaluator.py
    ├── loss_functions.py
    ├── models.py
    ├── layers.py
    ├── scoring.py
    └── video_dataset_loader.py
```

---

## 📜 Citation

If you use this work, please cite the paper:

```bibtex
@misc{lappas2024maskedskipunet,
  title={Masked Convolutions within Skip Connections for Video Anomaly Detection},
  author={Demetris Lappas and Dimitrios Makris and Vasileios Argyriou},
  year={2024},
  url={https://papers.ssrn.com/abstract=4768663}
}
```

---

## 🧐 Authors

* Demetris Lappas
* Dimitrios Makris
* Vasileios Argyriou

School of Computer Science and Mathematics, Kingston University London
