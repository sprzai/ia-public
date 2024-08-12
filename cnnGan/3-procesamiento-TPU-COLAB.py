import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging


# Configurar el registro de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'images/output'

# Definir el generador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Red generadora basada en bloques residuales
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        block1 = self.block1(x)
        residuals = self.residuals(block1)
        block2 = self.block2(residuals)
        block3 = self.block3(block1 + block2)
        block4 = self.block4(block3)
        return self.block5(block4)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

# Definir el discriminador
class Discriminator(nn.Module):
    def __init__(self, input_size=64):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Añadimos esta capa para asegurar un tamaño de salida fijo
        )
        
        self.final = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def conv_output_size(self, input_size):
        size = input_size
        for _ in range(4):  # Número de capas convolucionales
            size = (size - 1) // 2 + 1
        return size

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)  # Aplanar
        return self.final(x)

# Configuración del dataset y dataloader
def get_data_loader(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Entrenamiento de las redes
def train(generator, discriminator, dataloader, num_epochs=10, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)

            # Entrenar discriminador
            optimizer_d.zero_grad()
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)
            
            outputs = discriminator(imgs)
            d_loss_real = criterion(outputs, real_labels)
            
            noise = torch.randn(imgs.size(0), 3, 64, 64).to(device)
            fake_imgs = generator(noise)
            
            outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Entrenar generador
            optimizer_g.zero_grad()
            
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            
            g_loss.backward()
            optimizer_g.step()

            if (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                            f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Manejo de errores
try:
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    dataloader = get_data_loader()
    train(generator, discriminator, dataloader)
except Exception as e:
    logger.error(f'Error durante el entrenamiento: {e}')

logger.info('Entrenamiento completado.')

