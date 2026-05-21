import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Model (same idea as your CNNblocks) ----
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---- 2. Data ----
tfm = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True,  download=True, transform=tfm)
valid_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=64)

# ---- 3. Model, loss, optimizer, scheduler, logger ----
model     = CNN().to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
writer    = SummaryWriter(Path("runs/mnist").resolve())

# ---- 4. The training loop (this is what trainer.loop() does under the hood) ----
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = loss_fn(yhat, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct    += (yhat.argmax(1) == y).sum().item()
            n          += X.size(0)
    return total_loss / n, correct / n

EPOCHS = 5
best_val, patience, bad_epochs = float("inf"), 3, 0

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_dl, train=True)
    val_loss,   val_acc   = run_epoch(valid_dl, train=False)
    scheduler.step(val_loss)

    writer.add_scalars("loss",     {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars("accuracy", {"train": train_acc,  "val": val_acc},  epoch)
    print(f"Epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # early stopping
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping triggered")
            break

writer.close()