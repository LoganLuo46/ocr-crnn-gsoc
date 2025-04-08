import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.dataset import OCRDataset
from src.model import CRNN
from src.utils import encode_labels, decode_predictions, load_charset

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 10
IMG_HEIGHT = 32
IMG_WIDTH = 8192
MAX_LABEL_LEN = 1000  # Prevent overly long labels from causing CTC errors
CHARSET_PATH = "char_set.txt"


# Load character set
charset = load_charset(CHARSET_PATH)
char_to_idx = {char: idx for idx, char in enumerate(charset)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Load dataset
train_dataset = OCRDataset("data/images_ezcaray", "data/labels_ezcaray")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Check if charset covers all characters in the dataset
all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
all_chars = set("".join(all_labels))
missing_chars = sorted([ch for ch in all_chars if ch not in char_to_idx])

if missing_chars:
    print(f"Missing characters in charset.txt: {missing_chars}")
    print("Please add the missing characters to char_set.txt and rerun the script.")
    exit()

max_len = max([len(label) for label in all_labels])
print(f"Longest label length: {max_len}")

# Initialize model, loss function, and optimizer
model = CRNN(img_height=IMG_HEIGHT, num_classes=len(charset)).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_loss = float('inf')
patience = 2
wait = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_idx, (images, targets_text) in enumerate(train_loader):
        images = images.to(DEVICE)

        if isinstance(targets_text, tuple):
            targets_text = list(targets_text)

        clean_targets = []
        for t in targets_text:
            cleaned = ''.join([ch for ch in t if ch in char_to_idx])
            if 0 < len(cleaned) <= MAX_LABEL_LEN:
                clean_targets.append(cleaned)

        if not clean_targets:
            continue

        try:
            encoded_targets = [torch.tensor(encode_labels(t, char_to_idx), dtype=torch.long) for t in clean_targets]
            target_lengths = torch.tensor([len(t) for t in encoded_targets], dtype=torch.long)
        except Exception as e:
            print(f"Error encoding labels: {e}")
            continue

        flat_targets = torch.cat(encoded_targets).view(-1)
        outputs = model(images[:len(clean_targets)])
        log_probs = outputs.log_softmax(2).permute(1, 0, 2)
        input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)

        if any(target_lengths[i] > input_lengths[i] for i in range(len(target_lengths))):
            continue

        loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")

    # Early stopping logic
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

# Save the trained model
os.makedirs("checkpoints1", exist_ok=True)
torch.save(model.state_dict(), "checkpoints1/crnn_model.pth")
print("Model saved to checkpoints1/crnn_model.pth")