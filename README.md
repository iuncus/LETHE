# LETHE: Concept Erasure in Stable Diffusion

## Overview
LETHE is a research project focused on exploring the application of **concept erasure** in diffusion models to encourage abstraction in image generation. By selectively erasing low-level visual concepts—such as the color **red**—in the **CLIP** text encoder of **Stable Diffusion v1.5**, this process disrupts the model’s ability to interpret prompts in a conventional way, leading to progressively abstract outputs.

## Features
- **Concept Erasure in CLIP** – Modifies specific layers of CLIP to remove selected concepts.
- **Fine-Tuning Pipeline** – Implements concept erasure in Stable Diffusion models.
- **Dataset-Driven Learning** – Uses controlled monochromatic datasets for training.
- **Inference & Visualization** – Evaluates the impact of erasure on generated images.
- **Layer-Wise Analysis** – Maps CLIP layers to different abstraction levels.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch 1.12+
- Hugging Face `transformers` and `diffusers`
- OpenCLIP
- Jupyter Notebook (optional, for running `.ipynb` files)


## Project Structure
```
LETHE/
│── notebooks/
│   ├── CLIP_tune_baseline.ipynb  # Fine-tuning CLIP with concept erasure
│   ├── CLIP_layers.ipynb         # Lists and analyzes CLIP layers
│   ├── VAE.ipynb                 # Variational autoencoder exploration
│── scripts/
│   ├── infer_batch.py            # Inference script for abstract image generation
│   ├── utils.py                  # Helper functions
│── ds/
│   ├── all/                      # General dataset (blue, green, red images)
│   ├── any_red/                  # Subset of images with red
│   ├── multi_blue/               # Multi-shade blue dataset
│   ├── multi_red/                # Multi-shade red dataset
│   ├── not_blue/                 # Images excluding blue
│   ├── not_red/                  # Images excluding red
│── README.md                     # Project documentation
```

## Methodology
1. **Concept Erasure Implementation**
   - Identifies CLIP layers encoding specific visual features (e.g., red color).
   - Applies fine-tuning to minimize activation responses to the erased concept.
2. **Training Process**
   - Uses curated datasets of monochromatic images to enhance concept isolation.
   - Fine-tunes CLIP and measures abstraction effects at different layers.
   - Saves intermediate model checkpoints to assess the impact of erasure over time.
3. **Inference & Results**
   - Applies the modified models to Stable Diffusion for text-to-image generation.
   - Evaluates qualitative changes in output abstraction.
   - Compares model outputs before and after erasure.

## Key Findings
- Low-level concept erasure affects not just the target feature (e.g., color red) but also disrupts the model's broader ability to represent related concepts (e.g., apples).
- The abstraction process is influenced by the training dataset composition and **specific CLIP layers modified.
- Reproducibility remains a challenge, as identical training runs can yield visually distinct results.

## Usage

### Running Fine-Tuning
Run the notebook `CLIP_tune_baseline.ipynb` to fine-tune CLIP and remove a target concept.

### Running Inference
```bash
python scripts/infer_batch.py --model path/to/checkpoint --prompt "a red apple"
```
This generates images while testing the concept erasure effect.

## Future Work
- Extending beyond color to erase **shapes, textures, or object identities**.
- Refining CLIP layer selection for more precise abstraction control.
- Improving training stability and reproducibility.


