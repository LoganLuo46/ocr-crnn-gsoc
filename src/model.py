import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height=32, num_classes=80):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            

            nn.AdaptiveAvgPool2d((1, None)),  

        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        b, c, h, w = x.size()
        assert h == 1, "CNN output height must be 1 for CRNN"
        x = x.squeeze(2)  # Remove height dim ➜ [B, C, W]
        x = x.permute(0, 2, 1).contiguous()  # ➜ [B, W, C]

        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x


