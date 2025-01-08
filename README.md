## LETHE: Low-Level Concept Erasure For Abstract Image Generation
### Encouraging Stable Diffusion to deviate from prompt by erasing low-level concepts in the CLIP text encoder.

Training was done in CLIP_tune baseline.ipynb, saving a full updated CLIP model as checkpoint each epoch.

clip_layers.ipynb just to print all the individual layers in CLIP, helps deciding which parts to target.

infer_batch.py loads checkpoints from a training batch one at a time and performs inference.

ds folder contains the plurality of training data used in this project

