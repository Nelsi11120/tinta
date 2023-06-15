# -*- coding: utf-8 -*-
from torch import nn
from pathlib import Path

LABELS = Path("labels.txt").read_text().splitlines()

sketch_recognizer = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Linear(512, len(LABELS)),
)
