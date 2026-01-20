import numpy as np
import torch
import clip
from scipy.linalg import sqrtm
from PIL import Image
import torch.nn as nn


def calculate_clip_score(image_path, text_prompt, device):
    # Load model and tokenizer
    model, preprocess = clip.load('ViT-B/32', device=device) # TODO: Compare with metrics

    # Preprocess inputs
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text_prompt]).to(device)

    # Compute features and similarity
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate dot product
        score = (image_features @ text_features.T).item()

    return score


def calculate_fid(real_embeddings, gen_embeddings):
    # real_embeddings and gen_embeddings should be Numpy arrays of shape (N, 2048) 
    # extracted from an InceptionV3 model
    # Calculate mean and covariance
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2)

    # Calculate sqrt of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Final FID calculation
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid
