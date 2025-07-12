import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
import torch.nn.functional as F
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import rasterio.mask
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import torch.nn as nn

def get_gdf(path="Train_Data/Training_data_uhi_index_2025-02-18.csv"):
    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])],
        crs="EPSG:4326"
    ).to_crs("EPSG:2263")
    return gdf


def extract_LandCover_patch(data, point, patch_size_meters=100):

    geom = point.buffer(patch_size_meters * 3.28084)
    out_image, _ = rasterio.mask.mask(data, [geom], crop=True, nodata=0)

    return out_image[0]



def one_hot_encode_landcover(patch, all_classes=[1,2,3,4,5,6,7,8]):

    one_hot = np.zeros((len(all_classes), patch.shape[0], patch.shape[1]), dtype=np.uint8)

    for i, c in enumerate(all_classes):
        one_hot[i] = (patch == c).astype(np.uint8)

    return torch.tensor(one_hot, dtype=torch.float32)

class UHIPatchDataset(Dataset):
    def __init__(self, gdf, patch_size_meters=100, output_size=(132, 132)):
        self.gdf = gdf
        self.landcover_path = r"Referentiels/landcover_nyc_2021_6in_Clipped_5.tif" # Store the file path instead of an open handle in order to work with multiple Processes
        self.patch_size = patch_size_meters
        self.output_size = output_size

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        with rasterio.open(self.landcover_path) as data:
            row = self.gdf.iloc[idx]
            point = row.geometry
            label = row["UHI Index"]

            # Extract patch & one-hot encode it
            patch_xr = extract_LandCover_patch(data, point, self.patch_size)
            one_hot_patch = one_hot_encode_landcover(patch_xr, all_classes=[1,2,3,4,5,6,7,8])  # (C, H, W)

            patch_tensor = F.interpolate(
                one_hot_patch.unsqueeze(0),  # adding batch dimension => shape (1, C, H, W)
                size=self.output_size,
                mode='nearest'  # using 'nearest' for categorical data to avoid blending
            ).squeeze(0)  # shape => (8, 132, 132)

        return patch_tensor, torch.tensor(label-1, dtype=torch.float32)
    

class Adaptive_UHI_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Adaptive_UHI_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # From 200x200 to 100x100

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 100x100 to 50x50

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 50x50 to 25x25

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 25x25 to ~12x12
        )
        
        # New convolution layer with a larger kernel to capture global features.
        self.global_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3),  # Larger kernel
            nn.ReLU()
        )

        # Adaptive pooling to force output to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_conv(x)  # Apply the larger kernel conv for global features
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze()


def main():
    train_gdf = get_gdf(path="Train_Data/Training_data_uhi_index_2025-02-18.csv")


    landcover_data = rasterio.open(r"NYCOpenData/landcover_nyc_2021_6in_Clipped_5.tif")

    full_dataset = UHIPatchDataset(
                        gdf=train_gdf,
                        patch_size_meters=500,
                        output_size=(200,200)
                    )

    # Train/Val Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # Creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers= 4)


    model = Adaptive_UHI_CNN(num_classes=8)  
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    
    print("Starting Training: ")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        for patches, labels in train_loader:
            # Forward pass
            preds = model(patches).squeeze()  
            loss = criterion(preds, labels)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Compute R2 for training
        train_r2 = r2_score(train_labels, train_preds)

        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for patches, labels in val_loader:
                preds = model(patches).squeeze()
                loss = criterion(preds, labels)
                val_losses.append(loss.item())

                # Store predictions and labels for R2 computation
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Compute R2 for validation
        val_r2 = r2_score(val_labels, val_preds)
        # Adjust learning rate if R² is stable
        scheduler.step(val_r2)  # Reduce LR if R² plateaus

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {np.mean(train_losses):.8f}, Train R²: {train_r2:.8f}, "
            f"Val Loss: {np.mean(val_losses):.8f}, Val R²: {val_r2:.8f}")
        scheduler.step(np.mean(val_losses))


    # Saving after training is complete
    save_path = "_Models/DeepLR/LandCover_AdaptiveUHICNN_500m_240325.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



if __name__ == '__main__':
    main()
    
