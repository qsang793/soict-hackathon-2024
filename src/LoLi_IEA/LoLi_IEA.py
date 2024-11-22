import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.LoLi_IEA.colorSpace import HsvToRgb, RgbToHsv


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64 * 56 * 56, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(-1, 64 * 56 * 56)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x


class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=1, padding=0
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=1, padding=0
        )
        self.conv7 = nn.Conv2d(
            in_channels=128, out_channels=32, kernel_size=1, padding=0
        )
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        x4 = F.relu(self.conv4(x3))

        x5 = F.relu(self.conv5(x4))
        x6 = torch.cat([x3, x5], dim=1)

        x7 = F.relu(self.conv6(x6))
        x8 = torch.cat([x2, x7], dim=1)

        x9 = F.relu(self.conv7(x8))
        x10 = torch.cat([x1, x9], dim=1)

        x11 = F.relu(self.conv8(x10))

        return x11


class LoLi_IEA:
    def __init__(self, model_dir, device):
        self.device = device
        self.model_dir = model_dir

        # Initialize transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load models
        self.modelClass = Classification().to(self.device)
        self.modelClass.load_state_dict(
            torch.load(
                os.path.join(model_dir, "CLASSIFICATION.pt"), map_location=device
            )
        )

        self.modelValL = Enhance().to(self.device)
        self.modelValL.load_state_dict(
            torch.load(os.path.join(model_dir, "LOCAL.pt"), map_location=device)
        )
        self.modelValL.eval()

        self.modelValG = Enhance().to(self.device)
        self.modelValG.load_state_dict(
            torch.load(os.path.join(model_dir, "GLOBAL.pt"), map_location=device)
        )
        self.modelValG.eval()

    def enhance_image(self, image):
        """
        Enhance a single image

        Args:
            imgx (numpy.ndarray): Input image in RGB format

        Returns:
            numpy.ndarray: Enhanced image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rows, columns, dimension = image.shape

        # Convert to HSV
        HSVInput = RgbToHsv(image)

        # HSV Components
        hueComponent = HSVInput[:, :, 0]
        satComponent = HSVInput[:, :, 1]
        valComponent = HSVInput[:, :, 2]

        valComponentTensor = (
            torch.from_numpy(valComponent)
            .float()
            .to(self.device)
            .view(1, 1, rows, columns)
        )

        image = self.transform(Image.fromarray(image)).unsqueeze(0)

        self.modelClass.eval()
        with torch.no_grad():
            output = self.modelClass(image.to(self.device))
            _, predicted = torch.max(output.data, 1)
            if predicted == 1:
                valEnhancement = self.modelValL(valComponentTensor)
            else:
                valEnhancement = self.modelValG(valComponentTensor)

        rows1, columns1 = valEnhancement.shape[2:4]
        valEnhComponent = (
            valEnhancement.detach().cpu().numpy().reshape([rows1, columns1])
        )

        HSV = np.dstack((hueComponent, satComponent, valEnhComponent))
        algorithm = HsvToRgb(HSV)

        enhanced_image = np.dstack(
            (algorithm[:, :, 2], algorithm[:, :, 1], algorithm[:, :, 0])
        )
        enhanced_image = enhanced_image * 255
        return enhanced_image


if __name__ == "__main__":
    from tqdm import tqdm

    model_dir = "weights/LoLi_IEA"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoLi_IEA(model_dir, device)

    img_dir = "/home/manhckv/manhckv/soict/__temp/nighttime"

    save_dir = "__enhanced"
    os.makedirs(save_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(img_dir), desc="Enhancing images"):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        enhanced_image = model.enhance_image(image)

        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, enhanced_image)
