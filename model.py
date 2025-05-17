import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings("ignore")

# --- Settings ---
IMG_SIZE = 256  # ResNet expects 224x224
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_REAL_DIR = 'Dataset/Test/Real'
TRAIN_FAKE_DIR = 'Dataset/Test/Fake'
TEST_REAL_DIR = 'Dataset/Test/Real'
TEST_FAKE_DIR = 'Dataset/Test/Fake'

# --- Image Processing Functions ---
def compute_fft_features(img):
    """Extract frequency-domain features using FFT."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))
    return [
        float(np.mean(magnitude_spectrum)),
        float(np.std(magnitude_spectrum)),
        float(np.mean((magnitude_spectrum - np.mean(magnitude_spectrum)) ** 3) / (np.std(magnitude_spectrum) ** 3 + 1e-8))
    ]

def compute_edge_features(img):
    """Extract edge features using Canny edge detection."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    return [
        float(np.mean(edges)),
        float(np.std(edges))
    ]

def compute_lbp_features(img):
    """Extract texture features using Local Binary Patterns."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return [float(h) for h in hist[:3]]  # Use top 3 bins for simplicity

# --- Dataset ---
class DeepFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.features = []
        
        # Load real images (label 0)
        for file in os.listdir(real_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(real_dir, file)
                self._process_image(path, 0)
        
        # Load fake images (label 1)
        for file in os.listdir(fake_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(fake_dir, file)
                self._process_image(path, 1)
        
        print(f"Loaded {len(self.data)} images")

    def _process_image(self, path, label):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load {path}")
            return
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        fft_feats = compute_fft_features(img)
        edge_feats = compute_edge_features(img)
        lbp_feats = compute_lbp_features(img)
        features = fft_feats + edge_feats + lbp_feats
        
        if self.transform:
            img = self.transform(img)  # Returns tensor
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        self.data.append(img)
        self.features.append(features)
        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Image at index {idx} is not a tensor")
        return image, features, label

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Model ---
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        # Load pretrained ResNet-50
        self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        self.resnet.fc = nn.Identity()  # Remove final layer
        # Freeze early layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Feature processing for image processing features
        self.feature_norm = nn.BatchNorm1d(8)  # 3 FFT + 2 edge + 3 LBP
        self.feature_fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(2048 + 32, 512),  # ResNet-50 output + processed features
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, features):
        img_features = self.resnet(img)  # [batch, 2048]
        proc_features = self.feature_norm(features)
        proc_features = self.feature_fc(proc_features)  # [batch, 32]
        combined = torch.cat((img_features, proc_features), dim=1)
        return self.fc(combined)

# --- Training and Evaluation ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, feats, labels in loader:
        imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs, feats)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    losses = []
    preds, labels_list = [], []
    with torch.no_grad():
        for imgs, feats, labels in loader:
            imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs, feats)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            pred = torch.sigmoid(outputs)
            preds.extend(pred.cpu().numpy().flatten().tolist())
            labels_list.extend(labels.cpu().numpy().flatten().tolist())
            correct += ((pred > 0.5).float() == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    labels_array = np.array(labels_list)
    preds_array = np.array(preds)
    auc = roc_auc_score(labels_array, preds_array) if len(np.unique(labels_array)) > 1 else 0.0
    return np.mean(losses), accuracy, auc

# --- Main ---
if __name__ == "__main__":
    # Load datasets
    print("Loading datasets...")
    try:
        train_dataset = DeepFakeDataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, transform=train_transform)
        test_dataset = DeepFakeDataset(TEST_REAL_DIR, TEST_FAKE_DIR, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

    # Initialize model
    print("Initializing model...")
    try:
        model = DeepFakeDetector().to(DEVICE)
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Training loop
    print("Starting training...")
    best_auc = 0.0
    patience = 5
    early_stop_counter = 0
    for epoch in range(EPOCHS):
        try:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_auc = evaluate(model, test_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            
            scheduler.step(val_loss)
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), "deepfake_detector_best.pth")
                print("Model saved!")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            break

    # Final evaluation
    print("Final evaluation...")
    try:
        if os.path.exists("deepfake_detector_best.pth"):
            model.load_state_dict(torch.load("deepfake_detector_best.pth"))
            _, test_acc, test_auc = evaluate(model, test_loader, criterion, DEVICE)
            print(f"Test Accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
        else:
            print("No saved model found.")
    except Exception as e:
        print(f"Error in final evaluation: {e}")