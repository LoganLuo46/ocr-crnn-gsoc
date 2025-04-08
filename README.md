# CRNN for Historical OCR – GSoC 2025 RenAIssance Test II

This repository implements a deep learning-based Optical Character Recognition (OCR) system using a Convolutional Recurrent Neural Network (CRNN). The model is designed for early modern Spanish printed documents and serves as a submission for the GSoC 2025 RenAIssance project (Test II – OCR).

---

## Features

- CRNN model with CTC loss for sequence-to-sequence recognition
- Trained on wide, high-resolution grayscale images
- Works with early modern Spanish print layouts
- Includes both testing (with labels) and inference (without labels) modes
- Modular, readable PyTorch codebase

---

## Project Structure

```
.
├── train.py              # Script for training the CRNN model
├── test.py               # Evaluate model on labeled test images
├── inference.py          # Run inference on unlabeled document images
├── src/
│   ├── model.py          # CRNN model architecture (CNN + BiLSTM + CTC)
│   ├── dataset.py        # Custom OCRDataset loader for image-text pairs
│   └── utils.py          # Charset handling, encoding/decoding, metrics
├── checkpoints/
│   └── crnn_model.pth    # Trained PyTorch model weights
├── char_set.txt          # Character set used for encoding/decoding
└── requirements.txt      # Python dependencies

```
> Training data (`images_ezcaray`, `labels_ezcaray`) not included in the repo due to size and licensing concerns. Please contact the author if needed.



---

## Model Architecture

- **Input**: Grayscale images, resized to `32 × 8192` (H × W)
- **CNN**: Extracts spatial features (4 conv layers, batch norm, adaptive pooling)
- **RNN**: 2-layer bidirectional LSTM to capture sequence dependencies
- **CTC Decoder**: Enables flexible, alignment-free character sequence output

---

## Training Summary

- **Device**: CPU
- **Max label length**: 890 characters
- **Batch size**: 2  
- **Epochs**: Up to 10 (early stopping at epoch 8)
- **Loss function**: `CTCLoss(zero_infinity=True)`
- **Optimizer**: Adam (lr = 1e-3)

### Sample Training Logs

    Epoch 1/10 | Avg Loss: 6.8906  
    Epoch 2/10 | Avg Loss: 3.8652  
    Epoch 3/10 | Avg Loss: 3.3377  
    Epoch 4/10 | Avg Loss: 3.1279  
    Epoch 5/10 | Avg Loss: 3.1037  
    Epoch 6/10 | Avg Loss: 2.9740  
    Epoch 7/10 | Avg Loss: 2.9924  
    Epoch 8/10 | Avg Loss: 3.0094  
    Early stopping triggered at epoch 8.  


Model saved to `checkpoints/crnn_model.pth`.

---

