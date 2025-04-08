import os
import torch
from torch.utils.data import DataLoader
from src.dataset import OCRDataset
from src.model import CRNN
from src.utils import load_charset, decode_predictions
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, img_height, num_classes):
    """
    Load the trained CRNN model and set it to evaluation mode.
    """
    model = CRNN(img_height=img_height, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def compute_char_accuracy(pred_text, gt_text):
    """
    Compute a simple character-level accuracy.
    """
    if len(gt_text) == 0:
        return 0.0
    same_count = sum(1 for i in range(min(len(pred_text), len(gt_text))) if pred_text[i] == gt_text[i])
    return same_count / len(gt_text)

def main():
    print(f"Using device: {DEVICE}")
    CHARSET_PATH = "char_set.txt"
    MODEL_PATH = "checkpoints/crnn_model.pth"
    IMG_HEIGHT = 32  # Must match the height used during training

    # Load character set and build mapping dictionaries
    charset = load_charset(CHARSET_PATH)
    num_classes = len(charset)
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print("Loading trained model...")
    model = load_model(MODEL_PATH, img_height=IMG_HEIGHT, num_classes=num_classes)
    print("Model loaded.")

    # Build test data loader
    # Only image and label directories are required, same as during training
    test_dataset = OCRDataset("data/images_test", "data/labels_test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Running inference on test set...")
    os.makedirs("data/inference_result", exist_ok=True)

    total_acc = 0.0
    count = 0

    for idx, (images, gt_text) in enumerate(test_loader):
        # If gt_text is a list or tuple, take the first element (batch_size = 1)
        if isinstance(gt_text, (tuple, list)):
            gt_text = gt_text[0]
        images = images.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)  # Shape: (B, T, C)
            log_probs = outputs.log_softmax(2)
            preds = decode_predictions(log_probs, idx_to_char)
            pred_text = preds[0]

        acc = compute_char_accuracy(pred_text, gt_text)
        total_acc += acc
        count += 1

        print(f"Sample #{idx + 1}")
        print(f"Predicted: {pred_text}")
        print(f"Ground Truth: {gt_text}")
        print(f"Accuracy: {acc * 100:.2f}%")

        result_txt_path = os.path.join("data", "inference_result", f"result_{idx + 1}.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Predicted: {pred_text}\n")
            f.write(f"Ground Truth: {gt_text}\n")
            f.write(f"Accuracy: {acc * 100:.2f}%\n")

    avg_acc = total_acc / max(count, 1)
    print(f"\nDone. Average Accuracy: {avg_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
