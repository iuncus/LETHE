# LETHE: Low-Level Concept Erasure for Abstract Image Generation

## Overview
**LETHE** explores the application of **concept erasure** in diffusion models to encourage abstraction in image generation. By selectively erasing low-level visual concepts—such as the color **red**—in the **CLIP** text encoder of **Stable Diffusion v1.5**, this project disrupts the model’s ability to interpret prompts in a conventional way, leading to progressively abstract outputs.

## Project Structure
- **`CLIP_tune_baseline.ipynb`**: Fine-tuning notebook for CLIP. Saves a full updated model at each epoch.
- **`clip_layers.ipynb`**: Lists all CLIP layers, aiding in selecting which components to target for erasure.
- **`infer_batch.py`**: Loads trained checkpoints and performs inference to observe abstraction effects.
- **`ds/`**: Contains the dataset of monochromatic red images used for training.

## Concept & Methodology
1. **Concept Erasure**: We modify the **CLIP** text encoder to gradually remove its understanding of a fundamental visual concept.
2. **Training Process**: 
   - Fine-tuning is done on **Stable Diffusion v1.5** using a dataset of monochromatic red images.
   - Erasure is applied to different layers of CLIP to analyze their role in abstraction.
   - Updated models are saved at each epoch, allowing control over the degree of abstraction.

## Key Findings
- Low-level concept erasure affects not just the target feature (e.g., color red) but also disrupts the model's broader ability to represent related concepts (e.g., apples).
- The abstraction process is influenced by the training dataset composition and **specific CLIP layers modified.
- Reproducibility remains a challenge, as identical training runs can yield visually distinct results.

## Future Work
- Extending **concept erasure** beyond colors to other foundational concepts like **shapes and textures**.
- Mapping CLIP’s layers in greater detail to improve control over the abstraction process.
- Investigating ways to enhance reproducibility for creative and scientific applications.
