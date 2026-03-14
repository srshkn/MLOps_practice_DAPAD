import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import (
    DataLoaders,
    Path,
)

from lab2.architecture import SimpleCNN
from lab2.dataset import get_dataloaders

# Конфигурация
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_NAME = "model.pth"
SAVE_PATH = Path(__file__).resolve().parent / MODEL_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {MODEL_NAME}")


if __name__ == "__main__":
    main()
