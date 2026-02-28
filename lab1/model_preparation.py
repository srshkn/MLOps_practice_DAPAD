import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
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

# Конфигурация
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_NAME = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Модель
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


def get_dataloaders(path: str) -> DataLoaders:
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name="train", valid_name="test"),
        get_y=parent_label,
        item_tfms=Resize(128),
        batch_tfms=aug_transforms(size=128) + [Normalize.from_stats(*imagenet_stats)],
    )

    return data_block.dataloaders(path, bs=BATCH_SIZE)


def train_model(model: SimpleCNN, dls: DataLoaders, epochs: int, lr: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for xb, yb in dls.train:
            xb_clean = xb.as_subclass(torch.Tensor)
            yb_clean = yb.as_subclass(torch.Tensor).long()

            xb_clean, yb_clean = xb_clean.to(DEVICE), yb_clean.to(DEVICE)

            optimizer.zero_grad()

            preds = model(xb_clean)
            loss = criterion(preds, yb_clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb_clean.size(0)
            _, predicted = preds.max(1)
            correct += (predicted == yb_clean).sum().item()
            total += yb_clean.size(0)

        train_loss /= total
        train_acc = correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        )

    print("Training finished!")


def main():
    # Путь к данным для датасета
    path = Path(__file__).parent.resolve()

    # Инициализация датасета
    dls = get_dataloaders(path)

    # Инициализация модели
    model = SimpleCNN(num_classes=dls.c).to(DEVICE)

    # Обучение модели
    train_model(model, dls, EPOCHS, LEARNING_RATE)

    # Сохраняем модель в файл
    torch.save(model.state_dict(), MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}")


if __name__ == "__main__":
    main()
