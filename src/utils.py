def load_charset(path):
    """
    Load the character set from the specified file and return a list of characters.
    """
    with open(path, "r", encoding="utf-8") as f:
        return list(f.read())

def encode_labels(text, char_to_idx):
    """
    Convert a text string into a list of character indices.
    Characters not found in the dictionary are mapped to index 0 by default.
    """
    return [char_to_idx.get(ch, 0) for ch in text]

def decode_predictions(pred_indices, idx_to_char):
    """
    Decode predicted indices into their corresponding string representations.

    If pred_indices is a 3D tensor (e.g., shape [B, T, C]),
    it will first be converted to shape [B, T] using argmax over the last dimension.

    For each sequence:
      - Repeated consecutive characters are removed
      - Index 0 is treated as the CTC blank and ignored
      - Each valid index is mapped to a character using idx_to_char

    Args:
        pred_indices: Predicted index sequences, typically [B, T] or [B, T, C]
        idx_to_char: A dictionary mapping indices to characters (index 0 is blank)

    Returns:
        A single string if batch_size == 1, otherwise a list of strings.
    """
    if hasattr(pred_indices, "dim") and pred_indices.dim() == 3:
        pred_indices = pred_indices.argmax(dim=2)

    texts = []
    for indices in pred_indices:
        prev = None
        text = ''
        for idx in indices:
            idx_value = idx.item() if hasattr(idx, "item") else idx
            if idx_value != prev and idx_value != 0:
                text += idx_to_char.get(idx_value, '')
            prev = idx_value
        texts.append(text)

    return texts[0] if len(texts) == 1 else texts

def compute_accuracy(preds, targets):
    """
    Compute character-level accuracy by comparing predicted and ground truth strings.
    """
    correct = 0
    total = len(targets)
    for pred, target in zip(preds, targets):
        if pred == target:
            correct += 1
    return correct / total if total > 0 else 0
