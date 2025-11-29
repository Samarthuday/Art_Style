import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------------------------------------
# Device configuration
# -------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 224 if torch.cuda.is_available() else 128  # use small size if no GPU

# -------------------------------------------------------
# VGG Model for feature extraction
# -------------------------------------------------------
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # fewer layers to speed up
        self.chosen_features = ["0", "5", "10"]
        self.model = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT
        ).features[:29].eval()

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


class SoftClamp(nn.Module):
    def forward(self, image):
        # squashes values roughly into [0, 1] for nicer visualization
        return (torch.tanh(3 * image - 1.5) + 1) / 2.0


# -------------------------------------------------------
# Preprocessing and Deprocessing
# -------------------------------------------------------
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

unloader = transforms.Compose([
    SoftClamp(),
    transforms.ToPILImage()
])


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)


def save_image(tensor, file_name):
    folder = "./generated_images/"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    image = tensor.detach().cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(file_path)
    return file_path


def tensor_to_pil(tensor):
    image = tensor.detach().cpu().clone()
    if image.dim() == 4:
        image = image.squeeze(0)
    image = unloader(image)
    return image


# -------------------------------------------------------
# Style-transfer helpers
# -------------------------------------------------------
def gram_matrix(tensor):
    batch_size, channel, height, width = tensor.size()
    features = tensor.view(channel, height * width)
    G = torch.mm(features, features.t())
    return G / (channel * height * width)


def total_variation_loss(img):
    # encourages spatial smoothness
    x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


def tensor_to_numpy(img_tensor):
    """
    img_tensor: (1,3,H,W) or (3,H,W) in [0,1]
    returns: (H,W,3) float32 in [0,1]
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return img.astype(np.float32)


def compute_psnr_ssim(gen_img, ref_img):
    """
    Compute PSNR and SSIM between generated image and reference
    (here we use the content image as reference).
    """
    gen_np = tensor_to_numpy(gen_img)
    ref_np = tensor_to_numpy(ref_img)

    psnr_val = peak_signal_noise_ratio(ref_np, gen_np, data_range=1.0)
    ssim_val = structural_similarity(ref_np, gen_np, data_range=1.0, channel_axis=2)
    return float(psnr_val), float(ssim_val)


def run_style_transfer(
    vgg,
    content_img,
    style_img,
    alpha=1.0,
    beta=0.1,
    total_steps=500,
    tv_weight=0.0,
    track_history=False,
    history_interval=50,
):
    """
    Core style-transfer loop.
    Returns:
        generated_img (tensor),
        content_losses (list),
        style_losses (list),
        total_losses (list),
        history_images (list of tensors, optional)
    """
    vgg.to(device)

    generated_img = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated_img], lr=0.0005)

    content_features = vgg(content_img)
    style_features = vgg(style_img)

    content_losses = []
    style_losses = []
    total_losses = []
    history_images = []

    for step in range(total_steps):
        generated_features = vgg(generated_img)

        content_loss = 0
        style_loss = 0

        for gen_f, cont_f, sty_f in zip(
            generated_features, content_features, style_features
        ):
            content_loss += torch.mean((gen_f - cont_f) ** 2)

            G = gram_matrix(gen_f)
            A = gram_matrix(sty_f)
            style_loss += torch.mean((G - A) ** 2)

        tv_loss = tv_weight * total_variation_loss(generated_img)
        total_loss = alpha * content_loss + beta * style_loss + tv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        content_losses.append(content_loss.item())
        style_losses.append(style_loss.item())
        total_losses.append(total_loss.item())

        if track_history and (step % history_interval == 0 or step == total_steps - 1):
            history_images.append(generated_img.detach().clone())

    return generated_img.detach(), content_losses, style_losses, total_losses, history_images


def run_alpha_beta_experiment(
    vgg,
    content_img,
    style_img,
    alpha_values=(0.1, 1.0, 2.0),
    beta_values=(0.05, 0.1, 0.5),
    total_steps=300,
    tv_weight=0.0,
):
    """
    Runs multiple style-transfer configurations to show impact of alpha/beta.
    Returns a pandas DataFrame of results.
    """
    rows = []

    for alpha in alpha_values:
        for beta in beta_values:
            st.write(f"Running α={alpha}, β={beta} for {total_steps} steps...")
            gen_img, c_losses, s_losses, t_losses, _ = run_style_transfer(
                vgg,
                content_img,
                style_img,
                alpha=alpha,
                beta=beta,
                total_steps=total_steps,
                tv_weight=tv_weight,
                track_history=False,
            )

            final_content_loss = c_losses[-1]
            final_style_loss = s_losses[-1]
            final_total_loss = t_losses[-1]
            psnr_val, ssim_val = compute_psnr_ssim(gen_img, content_img)

            out_name = f"ab_sweep_a{alpha}_b{beta}.png"
            save_path = save_image(gen_img, out_name)

            rows.append({
                "alpha (content weight)": alpha,
                "beta (style weight)": beta,
                "final_content_loss": final_content_loss,
                "final_style_loss": final_style_loss,
                "final_total_loss": final_total_loss,
                "PSNR_vs_content": psnr_val,
                "SSIM_vs_content": ssim_val,
                "output_image_path": save_path,
            })

    df = pd.DataFrame(rows)
    return df


def run_tv_ablation(
    vgg,
    content_img,
    style_img,
    alpha=1.0,
    beta=0.1,
    total_steps=300,
    tv_weight=1e-6,
):
    """
    Compares results with and without TV loss.
    Returns:
        df_results, img_with_tv, img_without_tv (tensors)
    """
    configs = [
        {"name": "with_tv", "tv_weight": tv_weight},
        {"name": "without_tv", "tv_weight": 0.0},
    ]

    rows = []
    images = {}

    for cfg in configs:
        st.write(
            f"Running config '{cfg['name']}' with tv_weight={cfg['tv_weight']} "
            f"for {total_steps} steps..."
        )
        gen_img, c_losses, s_losses, t_losses, _ = run_style_transfer(
            vgg,
            content_img,
            style_img,
            alpha=alpha,
            beta=beta,
            total_steps=total_steps,
            tv_weight=cfg["tv_weight"],
            track_history=False,
        )

        final_content_loss = c_losses[-1]
        final_style_loss = s_losses[-1]
        final_total_loss = t_losses[-1]
        psnr_val, ssim_val = compute_psnr_ssim(gen_img, content_img)

        out_name = f"tv_ablation_{cfg['name']}.png"
        save_path = save_image(gen_img, out_name)

        rows.append({
            "config": cfg["name"],
            "tv_weight": cfg["tv_weight"],
            "final_content_loss": final_content_loss,
            "final_style_loss": final_style_loss,
            "final_total_loss": final_total_loss,
            "PSNR_vs_content": psnr_val,
            "SSIM_vs_content": ssim_val,
            "output_image_path": save_path,
        })

        images[cfg["name"]] = gen_img

    df = pd.DataFrame(rows)
    return df, images["with_tv"], images["without_tv"]


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("Neural Style Transfer")
st.write(
    "Upload your content and style images, configure hyperparameters, "
    "and explore different loss and metric settings."
)

# File uploader
content_file = st.file_uploader("Choose a content image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Choose a style image", type=["jpg", "png", "jpeg"])

# Hyperparameter inputs
alpha = st.slider("Alpha (Content Loss Weight)", 0.0, 2.0, 1.0, 0.1)
beta = st.slider("Beta (Style Loss Weight)", 0.0, 2.0, 0.1, 0.1)
total_steps = st.slider("Training Steps", 100, 5000, 500, 100)

tv_weight = st.slider(
    "TV Loss Weight (for smoothing, used in main run and TV ablation)",
    0.0,
    0.001,
    0.000001,
    0.000001,
    format="%.6f",
)

if content_file and style_file:
    content_img = load_image(content_file)
    style_img = load_image(style_file)

    st.subheader("Input Images")
    st.write("**Content Image:**")
    st.image(tensor_to_pil(content_img), use_column_width=True)

    st.write("**Style Image:**")
    st.image(tensor_to_pil(style_img), use_column_width=True)

    vgg = VGG().to(device)

    # -----------------------------------------------
    # MAIN STYLE TRANSFER RUN
    # -----------------------------------------------
    if st.button("Start Style Transfer"):
        st.write("Training in progress...")

        gen_img, content_losses, style_losses, total_losses, history_images = (
            run_style_transfer(
                vgg,
                content_img,
                style_img,
                alpha=alpha,
                beta=beta,
                total_steps=total_steps,
                tv_weight=tv_weight,
                track_history=True,
                history_interval=50,
            )
        )

        # Save and display final result
        file_path = save_image(gen_img, "generated_image.png")
        st.success(f"Style Transfer Complete! Image saved at: {file_path}")

        st.subheader("Final Generated Image")
        st.image(file_path, caption="Final Generated Image", use_column_width=True)

        # Compute quality metrics
        psnr_val, ssim_val = compute_psnr_ssim(gen_img, content_img)

        st.subheader("Objective Quality Metrics (vs Content Image)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PSNR (dB)", f"{psnr_val:.2f}")
        with col2:
            st.metric("SSIM", f"{ssim_val:.3f}")

        # Plot and display losses
        st.subheader("Loss Progression During Training")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(content_losses)), content_losses, label="Content Loss")
        ax.plot(range(len(style_losses)), style_losses, label="Style Loss")
        ax.plot(range(len(total_losses)), total_losses, label="Total Loss")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Variation During Style Transfer")
        ax.legend()
        st.pyplot(fig)

        # Show intermediate images
        if history_images:
            st.subheader("Intermediate Images (every ~50 steps)")
            pil_images = [tensor_to_pil(img) for img in history_images]
            captions = [
                f"Step {i * 50}" for i in range(len(pil_images))
            ]
            st.image(pil_images, caption=captions, width=300)

        # -------------------------------------------
        # Perceptual user rating survey (simple)
        # -------------------------------------------
        st.subheader("Perceptual User Rating (Optional)")
        st.write(
            "Rate how visually pleasing and stylistically rich the final image is. "
            "This is a simple subjective measure you can log for experiments."
        )
        rating = st.slider(
            "Overall quality rating (1 = poor, 5 = excellent)",
            min_value=1,
            max_value=5,
            value=3,
        )

        if st.button("Save Rating to CSV"):
            os.makedirs("./ratings/", exist_ok=True)
            ratings_path = "./ratings/user_ratings.csv"
            row = {
                "output_image_path": file_path,
                "alpha": alpha,
                "beta": beta,
                "tv_weight": tv_weight,
                "PSNR_vs_content": psnr_val,
                "SSIM_vs_content": ssim_val,
                "user_rating": rating,
            }

            if os.path.exists(ratings_path):
                existing = pd.read_csv(ratings_path)
                existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                existing.to_csv(ratings_path, index=False)
            else:
                pd.DataFrame([row]).to_csv(ratings_path, index=False)

            st.success(f"Rating saved to {ratings_path}")

    # -----------------------------------------------
    # α / β WEIGHT COMPARISON EXPERIMENT
    # -----------------------------------------------
    st.subheader("Content / Style Weight Comparison Experiment")
    st.write(
        "Run multiple combinations of α (content weight) and β (style weight) "
        "to see how they affect final losses and metrics."
    )

    default_alphas = [0.1, 1.0, 2.0]
    default_betas = [0.05, 0.1, 0.5]
    st.write(f"Using α values: {default_alphas}")
    st.write(f"Using β values: {default_betas}")

    if st.button("Run α / β Sweep"):
        df_ab = run_alpha_beta_experiment(
            vgg,
            content_img,
            style_img,
            alpha_values=default_alphas,
            beta_values=default_betas,
            total_steps=min(300, total_steps),
            tv_weight=tv_weight,
        )
        st.write("Results (one row per α/β combination):")
        st.dataframe(df_ab)

    # -----------------------------------------------
    # TV LOSS ABLATION EXPERIMENT
    # -----------------------------------------------
    st.subheader("Total Variation (TV) Loss Ablation")
    st.write(
        "Compare results with and without TV loss using the same α, β, and steps. "
        "TV loss often reduces noise and artifacts at the cost of some sharpness."
    )

    if st.button("Run TV Loss Ablation Experiment"):
        df_tv, img_with_tv, img_without_tv = run_tv_ablation(
            vgg,
            content_img,
            style_img,
            alpha=alpha,
            beta=beta,
            total_steps=min(300, total_steps),
            tv_weight=tv_weight if tv_weight > 0 else 1e-6,
        )

        st.write("Ablation Results Table:")
        st.dataframe(df_tv)

        col1, col2 = st.columns(2)
        with col1:
            st.write("With TV Loss")
            st.image(tensor_to_pil(img_with_tv), use_column_width=True)
        with col2:
            st.write("Without TV Loss")
            st.image(tensor_to_pil(img_without_tv), use_column_width=True)

else:
    st.info("Please upload both a content image and a style image to begin.")
