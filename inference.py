import os
import torch
from PIL import Image
from torchvision import transforms
from src.model import CRNN
from src.utils import load_charset, decode_predictions

# ===== Model and path configuration =====
IMG_HEIGHT = 32
IMG_WIDTH = 8192
CHARSET_PATH = "char_set.txt"
MODEL_PATH = "checkpoints/crnn_model.pth"
IMAGE_DIR = "data/images_ezcaray"
OUTPUT_DIR = "data/inference_result"

# ===== Load charset =====
charset = load_charset(CHARSET_PATH)
idx_to_char = {i: c for i, c in enumerate(charset)}

# ===== Initialize model =====
model = CRNN(img_height=IMG_HEIGHT, num_classes=len(charset))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ===== Image preprocessing =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Start inference =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.endswith(".jpg"):
        continue

    image_path = os.path.join(IMAGE_DIR, fname)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Shape: [1, 1, H, W]

    with torch.no_grad():
        output = model(image)                      # Shape: [1, T, C]
        probs = output.log_softmax(2)              # Shape: [1, T, C]
        pred = torch.argmax(probs, dim=2)          # Shape: [1, T]
        pred_indices = pred.cpu().numpy().tolist() # [[...]]
        text = decode_predictions(pred_indices, idx_to_char)

    out_path = os.path.join(OUTPUT_DIR, fname.replace(".jpg", ".txt"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved prediction for {fname}")
