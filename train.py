import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# ─── CONFIG ───────────────────────────────────────────
DATASET_PATH = 'project/PlantVillage'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
NUM_CLASSES = 15
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'crop_disease_model.pth'
# ──────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}\n")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # ✅ num_workers=0 fixes Windows multiprocessing error
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"🖼️  Training: {train_size} | Validation: {val_size}\n")

    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)
    print("✅ MobileNetV2 loaded\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

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

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {val_loss:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_SAVE_PATH)
            print(f"  💾 Saved! Best Val Acc: {val_acc:.1f}%")

    # Fine tuning
    print("\n🔧 Fine-tuning last layers...")
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

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
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, MODEL_SAVE_PATH)
            print(f"  💾 New best: {ft_acc:.1f}%")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('training_results.png')
    plt.show()

    print(f"\n✅ Done! Best Accuracy: {best_val_acc:.1f}%")
    print(f"💾 Model saved: {MODEL_SAVE_PATH}")
    print(f"📊 Graph saved: training_results.png")


# ✅ Required for Windows multiprocessing
if __name__ == '__main__':
    main()