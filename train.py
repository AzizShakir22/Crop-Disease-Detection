import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

# ─── CONFIG ───────────────────────────────────────────
DATASET_PATH = 'project/PlantVillage'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
NUM_CLASSES = 15
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'crop_disease_model.pth'
FULL_MODEL_PATH = 'crop_disease_full.pth'
# ──────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}\n")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    class_names = full_dataset.classes
    print(f"📁 Classes found ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print(f"\n📊 Total images: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"🖼️  Training: {train_size} | Validation: {val_size}\n")

    # ─── MODEL ────────────────────────────────────────
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # Unfreeze ALL layers from the start
    for param in model.parameters():
        param.requires_grad = True

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)
    print("✅ MobileNetV2 loaded — all layers trainable\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    train_accs, val_accs = [], []
    best_val_acc = 0.0

    print("=" * 55)
    print("🌿 Training Started!")
    print("=" * 55)

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        elapsed = time.time() - start

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"Loss: {val_loss:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save state dict version
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'val_accuracy': val_acc,
                'num_classes': NUM_CLASSES
            }, MODEL_SAVE_PATH)
            print(f"  💾 Saved! Best Val Acc: {val_acc:.1f}%")

    # ─── FINE TUNING ──────────────────────────────────
    print("\n🔧 Fine-tuning all layers...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        ft_acc = 100. * val_correct / val_total
        print(f"Fine-tune [{epoch+1}/5] | Val Acc: {ft_acc:.1f}%")

        if ft_acc > best_val_acc:
            best_val_acc = ft_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'val_accuracy': ft_acc,
                'num_classes': NUM_CLASSES
            }, MODEL_SAVE_PATH)
            print(f"  💾 New best: {ft_acc:.1f}%")

    # ─── FINAL SAVE BOTH FORMATS ──────────────────────
    print("\n💾 Saving final complete models...")

    # Make sure all params are included
    for param in model.parameters():
        param.requires_grad = True

    # Format 1: Full model (larger, easier to load)
    torch.save(model, FULL_MODEL_PATH)

    # Format 2: State dict (smaller, more portable)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'val_accuracy': best_val_acc,
        'num_classes': NUM_CLASSES
    }, MODEL_SAVE_PATH)

    # ─── VERIFY FILE SIZES ────────────────────────────
    size_full  = os.path.getsize(FULL_MODEL_PATH) / (1024 * 1024)
    size_state = os.path.getsize(MODEL_SAVE_PATH)  / (1024 * 1024)

    print(f"\n📦 Full model  ({FULL_MODEL_PATH}):  {size_full:.1f} MB")
    print(f"📦 State dict  ({MODEL_SAVE_PATH}): {size_state:.1f} MB")

    if size_full > 30:
        print("✅ Full model size looks correct!")
    else:
        print("⚠️  Full model still small — check training.")

    # ─── PLOT ─────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs,   label='Val Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"💾 Upload to Hugging Face → {FULL_MODEL_PATH}")
    print(f"📊 Graph saved: training_results.png")


# ✅ Required for Windows
if __name__ == '__main__':
    main()