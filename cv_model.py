import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# --- Configuration ---
DATA_DIR = "data"     
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TABULAR_FILE = os.path.join(DATA_DIR, "welding_data.csv")
MODEL_CV_PATH = os.path.join("models", "cnn_defect_model.pth")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# --- Dataset Class ---
class WeldDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Create mapping for defect types
        self.defect_labels = self.data['DefectType'].unique()
        self.label_map = {label: idx for idx, label in enumerate(self.defect_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['ImageID'])
        image = Image.open(img_name).convert('L') # Gray scale
        
        defect_str = self.data.iloc[idx]['DefectType']
        label = self.label_map[defect_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Model Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128) # Input image 64x64 -> 16x16 after 2 pools
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cv_model():
    print("Initializing CV Model Training...")
    
    if not os.path.exists(TABULAR_FILE) or not os.path.exists(IMAGES_DIR):
         print(f"Error: Data not found. Run data_gen.py first.")
         return

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset & DataLoader
    dataset = WeldDataset(TABULAR_FILE, IMAGES_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(dataset.defect_labels)
    print(f"Detected {num_classes} defect classes: {dataset.defect_labels}")

    # Model, Loss, Optimizer
    model = SimpleCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    # Save Model
    print("Saving CV Model...")
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': dataset.label_map,
        'idx_to_label': dataset.idx_to_label
    }, MODEL_CV_PATH)
    print("CV Model saved.")

if __name__ == "__main__":
    train_cv_model()
