import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 224 if torch.cuda.is_available() else 128  # use small size if no GPU

# VGG Model for feature extraction
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10"]  # Use fewer layers to speed up
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29].eval()  # pretrained model
    
    def forward(self, x):
        features = []  # list of features
        for layer_num, layer in enumerate(self.model):
            x = layer(x)  # forward pass
            if str(layer_num) in self.chosen_features:  # if layer is in chosen features
                features.append(x)
        return features
    
class SoftClamp(nn.Module):
    def forward(self, image):
        return (torch.tanh(3 * image - 1.5) + 1) / 2.0

# Preprocessing and Deprocessing
def load_image(image):
    image = Image.open(image).convert("RGB")  # Open image and ensure it's RGB
    image = loader(image).unsqueeze(0)  # add the batch dimension
    return image.to(device)  # use GPU if available

def save_image(tensor, file_name):
    folder = "./generated_images/"
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    file_path = os.path.join(folder, file_name)  # Full path for the file
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)  # convert tensor to PIL image
    image.save(file_path)
    return file_path

def show_image(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)  # convert tensor to PIL Image
    st.image(image, use_column_width=True)

# Image transformations (similar to PyTorch approach)
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # resize image
    transforms.ToTensor(),  # convert image to Tensor
])

unloader = transforms.Compose([
    SoftClamp(),  # clamp image values for better visualization
    transforms.ToPILImage()  # convert Tensor to PIL Image
])

# Function to calculate Gram matrix for style loss
def gram_matrix(tensor):
    batch_size, channel, height, width = tensor.size()
    features = tensor.view(channel, height * width)
    G = torch.mm(features, features.t())  # Gram matrix
    return G / (channel * height * width)

# Streamlit UI
st.title("Neural Style Transfer")
st.write("Upload your content and style images, and configure the hyperparameters for training.")

# File uploader
content_file = st.file_uploader("Choose a content image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Choose a style image", type=["jpg", "png", "jpeg"])

# Hyperparameter inputs
alpha = st.slider("Alpha (Content Loss Weight)", 0.0, 2.0, 1.0, 0.1)
beta = st.slider("Beta (Style Loss Weight)", 0.0, 2.0, 0.1, 0.1)
total_steps = st.slider("Training Steps", 100, 5000, 500, 100)  # Lower steps for faster processing

# Show uploaded images
if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file)

    st.write("Content Image:")
    show_image(content_img)

    st.write("Style Image:")
    show_image(style_img)

    # Initialize generated image (start with content image)
    generated_img = content_img.clone()  # Initialize generated image as content image
    generated_img.requires_grad_(True)  # Enable gradient computation

    # Hyperparameters and optimizer
    optimizer = torch.optim.Adam([generated_img], lr=0.0005)

    vgg = VGG().to(device)  # Pretrained VGG19 model

    # Get style and content features
    content_features = vgg(content_img)
    style_features = vgg(style_img)

    # Store all generated images for display at the end
    all_generated_images = []

    # Training loop with loss tracking
    if st.button("Start Style Transfer"):
        st.write("Training in progress...")

        content_losses = []  # Track content loss
        style_losses = []  # Track style loss
        total_losses = []  # Track total loss

        for step in tqdm(range(total_steps)):
            generated_img_features = vgg(generated_img)  # Features of generated image
            content_img_features = vgg(content_img)
            style_img_features = vgg(style_img)

            style_loss = content_loss = 0
            # Loop through all the features
            for gen_feature, content_feature, style_feature in zip(generated_img_features, content_img_features, style_img_features):
                batch_size, channel, height, width = gen_feature.shape  # Batch size = 1
                content_loss += torch.mean((gen_feature - content_feature) ** 2)  # Content loss

                # Compute Gram Matrix
                G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
                A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
                style_loss += torch.mean((G - A) ** 2)  # Style loss

            # Compute total loss
            total_loss = alpha * content_loss + beta * style_loss

            # Log losses
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
            total_losses.append(total_loss.item())

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the image

            if step % 50 == 0:  # Store intermediate image every 50 steps
                all_generated_images.append(generated_img.cpu().clone())

        # Save final generated image
        file_path = save_image(generated_img, "generated_image.png")
        st.write(f"Style Transfer Complete! The image is saved at: {file_path}")

        # Display the final generated image
        st.image(file_path, caption="Final Generated Image", use_column_width=True)

        # Plot and display the losses
        st.write("Loss Progression During Training")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(total_steps), content_losses, label="Content Loss", color="blue")
        ax.plot(range(total_steps), style_losses, label="Style Loss", color="orange")
        ax.plot(range(total_steps), total_losses, label="Total Loss", color="green")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Variation During Style Transfer")
        ax.legend()
        st.pyplot(fig)

        # Show intermediate images
        st.write("Intermediate Images (every 50 steps):")
        images_to_display = []
        for img in all_generated_images:
            images_to_display.append(unloader(img.squeeze(0)))  # Convert tensors to PIL images

        st.image(images_to_display, width=300, caption=[f"Step {i * 50}" for i in range(len(images_to_display))])
