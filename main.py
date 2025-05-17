import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# --- Settings ---
IMG_SIZE = 256  # Model input size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deepfake_detector_best.pth"

# --- Image Processing Functions ---
def compute_fft_features(img):
    """Extract frequency-domain features and FFT visualization."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))
    features = [
        float(np.mean(magnitude_spectrum)),
        float(np.std(magnitude_spectrum)),
        float(np.mean((magnitude_spectrum - np.mean(magnitude_spectrum)) ** 3) / (np.std(magnitude_spectrum) ** 3 + 1e-8))
    ]
    return features, magnitude_spectrum

def compute_edge_features(img):
    """Extract edge features and Canny edge map."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    features = [
        float(np.mean(edges)),
        float(np.std(edges))
    ]
    return features, edges

def compute_lbp_features(img):
    """Extract LBP features and LBP visualization."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    features = [float(h) for h in hist[:3]]
    return features, lbp

def preprocess_image(image):
    """Preprocess image to 224x224 while handling any input size."""
    img = np.array(image.convert('RGB'))  # PIL to numpy (RGB)
    # Resize with aspect ratio
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    # Pad to 224x224
    pad_h = (IMG_SIZE - new_h) // 2
    pad_w = (IMG_SIZE - new_w) // 2
    padded_img = cv2.copyMakeBorder(img, pad_h, IMG_SIZE - new_h - pad_h, pad_w, IMG_SIZE - new_w - pad_w,
                                    cv2.BORDER_CONSTANT, value=0)
    return padded_img

# --- Model Definition ---
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.resnet = models.resnet50(weights=None)  # Weights loaded from .pth
        self.resnet.fc = nn.Identity()
        self.feature_norm = nn.BatchNorm1d(8)  # 3 FFT + 2 edge + 3 LBP
        self.feature_fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, features):
        img_features = self.resnet(img)
        proc_features = self.feature_norm(features)
        proc_features = self.feature_fc(proc_features)
        combined = torch.cat((img_features, proc_features), dim=1)
        return self.fc(combined)

# --- Load Model from .pth ---
@st.cache_resource
def load_model():
    try:
        model = DeepFakeDetector().to(DEVICE)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            return model
        else:
            st.error(f"Model file '{MODEL_PATH}' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Prediction Function ---
def predict_image(model, img, features):
    """Predict real vs. fake probability."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor, feat_tensor)
        prob = torch.sigmoid(output).item()
    
    return prob

# --- Visualization Functions ---
def plot_feature_distribution(features):
    """Bar chart of feature values."""
    labels = ['FFT Mean', 'FFT Std', 'FFT Skew', 'Edge Mean', 'Edge Std', 'LBP1', 'LBP2', 'LBP3']
    fig = px.bar(x=labels, y=features, title="Feature Values",
                 labels={'x': 'Feature', 'y': 'Value'},
                 color=labels, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(showlegend=False)
    return fig

def plot_prediction_confidence(prob):
    """Bar chart of prediction probabilities."""
    labels = ['Real', 'Fake']
    values = [1 - prob, prob]
    colors = ['#1f77b4', '#ff7f0e']
    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colors)
    ])
    fig.update_layout(title="Prediction Confidence",
                      yaxis_title="Probability",
                      yaxis_range=[0, 1],
                      template="plotly_white")
    return fig

def plot_feature_importance(features):
    """Simulated feature importance (normalized magnitudes)."""
    labels = ['FFT Mean', 'FFT Std', 'FFT Skew', 'Edge Mean', 'Edge Std', 'LBP1', 'LBP2', 'LBP3']
    importance = np.abs(features) / (np.max(np.abs(features)) + 1e-8)
    fig = px.bar(x=labels, y=importance, title="Feature Importance (Simulated)",
                 labels={'x': 'Feature', 'y': 'Relative Importance'},
                 color=labels, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(showlegend=False)
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title("Deepfake Detection App")
st.markdown("""
Upload an image to analyze its FFT, Canny edge, and LBP features, and determine if it's real or fake.
The app uses a pretrained ResNet-50 model with image processing features for detection.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"], help="Supports JPG, PNG, JPEG files")

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    with st.spinner("Processing image features..."):
        img = preprocess_image(image)
        
        # Compute features
        fft_feats, fft_img = compute_fft_features(img)
        edge_feats, edge_img = compute_edge_features(img)
        lbp_feats, lbp_img = compute_lbp_features(img)
        all_features = fft_feats + edge_feats + lbp_feats
        
        # Normalize visualizations
        fft_img = (fft_img - fft_img.min()) / (fft_img.max() - fft_img.min() + 1e-8)
        edge_img = edge_img / 255.0
        lbp_img = (lbp_img - lbp_img.min()) / (lbp_img.max() - lbp_img.min() + 1e-8)
    
    # Display features
    st.header("Image Processing Features")
    st.markdown("These features help the model detect deepfake artifacts.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(fft_img, caption="FFT Magnitude Spectrum", use_column_width=True)
        st.markdown("Shows frequency components, useful for detecting artificial patterns.")
    with col2:
        st.image(edge_img, caption="Canny Edge Map", use_column_width=True)
        st.markdown("Highlights structural edges, revealing inconsistencies.")
    with col3:
        st.image(lbp_img, caption="LBP Texture", use_column_width=True)
        st.markdown("Captures local texture patterns, sensitive to subtle manipulations.")
    
    # Load model and predict
    model = load_model()
    if model is not None:
        with st.spinner("Analyzing with deepfake detection model..."):
            prob = predict_image(model, img, all_features)
            label = "Fake" if prob > 0.5 else "Real"
            confidence = prob if prob > 0.5 else 1 - prob
            
            # Display prediction
            st.header("Prediction Result")
            st.markdown(f"**{label}** (Confidence: **{confidence:.2%}**)")
            st.progress(confidence)
            
            # Display analysis
            st.header("Model Analysis")
            st.markdown("Visualizing how the model evaluates the image.")
            
            # Prediction confidence
            st.subheader("Prediction Confidence")
            st.plotly_chart(plot_prediction_confidence(prob), use_container_width=True)
            
            # Feature distribution
            st.subheader("Feature Distribution")
            st.plotly_chart(plot_feature_distribution(all_features), use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance (Simulated)")
            st.markdown("Shows relative contribution of each feature (based on normalized magnitudes).")
            st.plotly_chart(plot_feature_importance(all_features), use_container_width=True)
else:
    st.info("Please upload an image to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch | Deepfake Detection Model using ResNet-50")