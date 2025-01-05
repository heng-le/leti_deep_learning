# LETI 2024 Summer Project: Base Editing Prediction

## Overview
This project focuses on leveraging advanced machine learning techniques to predict outcomes for **base editing**. The primary objectives include:
1. **Feature extraction**: Identifying the most relevant features for base editing.
2. **Model development**: Creating and evaluating predictive models to forecast base editing efficiency.

## Methods
This project employs a combination of cutting-edge deep learning and feature extraction techniques, including:

- **Convolutional Neural Networks (CNNs)**:
  - Used to analyze sequence data and detect spatial patterns related to base editing.
  - Enabled efficient feature extraction from complex nucleotide sequences.

- **Autoencoders**:
  - Applied for dimensionality reduction and unsupervised feature learning.
  - Extracted latent features that highlight key properties of input data while reducing noise.

- **Graph Neural Networks (GNNs)**:
  - Designed to model sequence-structure relationships, capturing interactions in RNA or DNA secondary structure.
  - Provided insights into connectivity and dependencies within biological datasets.

## Dataset
- **Nucleotide sequences** annotated with base editing efficiency labels.
- Features include PAM compatibility, editing positions, and nucleotide context.
- Both raw and encoded sequence data were used for training and validation.

