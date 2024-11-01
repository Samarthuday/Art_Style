

# Art Style Transfer with Convolutional Neural Networks

This project implements a convolutional neural network (CNN) for art style transfer. The code leverages TensorFlow and Keras to train a model using content images and style images to create stylized output. The content images are downloaded from WikiArt, and the style images are sourced from Kaggle.

## Table of Contents
- [Art Style Transfer with Convolutional Neural Networks](#art-style-transfer-with-convolutional-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
    - [Folder Structure:](#folder-structure)
  - [Installation](#installation)
    - [Prerequisites:](#prerequisites)
    - [Install dependencies:](#install-dependencies)
  - [Usage](#usage)
    - [Example:](#example)
  - [Model](#model)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results Visualization](#results-visualization)
  - [Acknowledgments](#acknowledgments)

## Overview
Art style transfer is a technique where the style of one image (e.g., a painting) is applied to the content of another image (e.g., a photograph). In this project, we use CNNs to blend the artistic style of one image with the content of another.

The architecture involves:
- Data loading and preprocessing
- A simple CNN model for classification (this can be expanded for style transfer)
- Data augmentation to improve model generalization
- Training the model on content images and applying style transfer techniques for visual outputs

## Dataset
- **Content Images**: Downloaded from [WikiArt](https://www.wikiart.org/)
- **Style Images**: Downloaded from [Kaggle](https://www.kaggle.com/)
  
### Folder Structure:
```
.
├── Artworks/               # Folder for style images
├── W0/                     # Folder for content images (first batch)
├── W19/                    # Folder for content images (second batch)
```

## Installation

### Prerequisites:
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- tqdm (for progress bars)

### Install dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn tqdm
```

## Usage

1. Clone the repository or download the code.
2. Place your content and style images in the appropriate folders (`W0`, `W19` for content and `Artworks` for style).
3. Run the code in your favorite IDE or Jupyter Notebook.

### Example:
```bash
python art_style_transfer.py
```

## Model
A simple CNN model with the following architecture:
- **Conv2D layers** with ReLU activation
- **MaxPooling** for downsampling
- **Dense layers** for classification with a dropout layer for regularization

The model can be further adapted to support actual style transfer using Gram matrix computation or other neural style transfer techniques.

## Training
The content images are split into training and testing sets (80-20 split). Data augmentation is applied to prevent overfitting. The model is trained using the Adam optimizer with categorical cross-entropy as the loss function.

Key parameters:
- **Epochs**: 10 (can be modified)
- **Batch Size**: 32

The model is trained using content images, and dummy labels are used for simplicity. For real-world style transfer, style and content loss would be calculated.

## Evaluation
After training, the model is evaluated using the testing data, and a classification report is generated to display metrics like precision, recall, and F1 score.

## Results Visualization
The code includes a `visualize_content_style_output()` function to display content images, style images, and the stylized output. In this demo, a simple blend of content and style images is shown.

To visualize the output:
- Content, style, and stylized images are shown side by side.
- The number of samples to display can be adjusted.

## Acknowledgments
- Content images are sourced from [WikiArt](https://www.wikiart.org/).
- Style images are sourced from [Kaggle](https://www.kaggle.com/).
- The implementation is inspired by the paper *Artistic Style Transfer with Convolutional Neural Network*.

