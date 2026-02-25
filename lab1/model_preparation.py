import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    GrandparentSplitter,
    ImageBlock,
    Normalize,
    Path,
    Resize,
    aug_transforms,
    get_image_files,
    imagenet_stats,
    parent_label,
)

path = Path(__file__).resolve().parent.parent / "train"


data_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name="train", valid_name="test"),
    get_y=parent_label,
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(size=128) + [Normalize.from_stats(*imagenet_stats)],
)

dls = data_block.dataloaders(path, bs=32)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = SimpleCNN(num_classes=dls.c)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
best_acc = 0.0

for epoch in range(epochs):
    # ---- TRAIN ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for xb, yb in dls.train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        _, predicted = preds.max(1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    train_loss /= total
    train_acc = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
    )

print("Training finished!")

torch.save(model.state_dict(), "model.pth")

print("Model saved to model.pkl")
