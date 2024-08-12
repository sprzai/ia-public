import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import time
from tqdm import tqdm

# Configurar el registro de logs
log_file = 'gan_checkpoint_epoch.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
dataset_path = 'images/output'

# Definir el generador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Entrada es Z, va a un convtranspose
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Estado: (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Estado: (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Estado: (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Estado: (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Estado final: (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Definir el discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input es (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (64) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (128) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (256) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Estado: (512) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Configuración del dataset y dataloader
def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dataloader

def save_models(generator, discriminator, optimizer_g, optimizer_d, epoch, path='models'):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_path = f'{path}/gan_checkpoint_epoch_{epoch}.pth'
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'epoch': epoch
        }, save_path)

        logger.info(f'Model saved to {save_path}')
    except Exception as e:
        logger.error(f'Error saving models: {e}')

def generate_and_save_images(generator, epoch, num_images=16, path='generated_images'):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_images, 100, 1, 1, device=device)
            generated_images = generator(noise).cpu()
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = generated_images[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Desnormalizar
            ax.imshow(img)
            ax.axis('off')
        
        save_path = f'{path}/generated_images_epoch_{epoch}.png'
        plt.savefig(save_path)
        plt.close()
        logger.info(f'Generated images saved to {save_path}')
    except Exception as e:
        logger.error(f'Error generating and saving images: {e}')

@torch.no_grad()
def calculate_fid_like_score(real_images, generated_images, batch_size=32):
    logger.info("Calculating FID-like score...")
    try:
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_model.eval()

        def get_features(images):
            features = []
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size].to(device)
                feat = inception_model(batch)[0].view(batch.shape[0], -1)
                features.append(feat.cpu().numpy())
            return np.concatenate(features, axis=0)

        real_features = get_features(real_images)
        gen_features = get_features(generated_images)

        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
        logger.info(f"FID-like score calculated: {fid}")
        return fid
    except Exception as e:
        logger.error(f'Error calculating FID-like score: {e}')
        return float('inf')

@torch.no_grad()
def evaluate_model(generator, dataloader, epoch, num_images=1000):
    logger.info(f"Evaluating model at epoch {epoch}...")
    try:
        generator.eval()
        real_images = next(iter(dataloader))[0][:num_images]
        
        noise = torch.randn(num_images, 100, 1, 1, device=device)
        generated_images = generator(noise)

        fid_score = calculate_fid_like_score(real_images, generated_images)
        logger.info(f'Epoch {epoch}: FID-like score: {fid_score}')

        generate_and_save_images(generator, epoch)

        return fid_score
    except Exception as e:
        logger.error(f'Error evaluating model: {e}')
        return float('inf')

def train(generator, discriminator, dataloader, num_epochs=10, lr=0.0002, beta1=0.5, save_interval=10, eval_interval=5):
    logger.info("Starting training process...")
    try:
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(64, 100, 1, 1, device=device)
        real_label = 1.0
        fake_label = 0.0

        optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        best_fid = float('inf')
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            generator.train()
            discriminator.train()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for i, (real_images, _) in enumerate(pbar):
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Entrenar discriminador
                optimizer_d.zero_grad()
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                output = discriminator(real_images)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake = generator(noise)
                label.fill_(fake_label)
                output = discriminator(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizer_d.step()

                # Entrenar generador
                optimizer_g.zero_grad()
                label.fill_(real_label)
                output = discriminator(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizer_g.step()

                pbar.set_postfix({'D_loss': errD.item(), 'G_loss': errG.item(), 'D(x)': D_x, 'D(G(z))': D_G_z2})

            epoch_end_time = time.time()
            logger.info(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

            # Evaluar y guardar el modelo periódicamente
            if (epoch + 1) % eval_interval == 0:
                fid_score = evaluate_model(generator, dataloader, epoch + 1)
                if fid_score < best_fid:
                    best_fid = fid_score
                    save_models(generator, discriminator, optimizer_g, optimizer_d, epoch + 1, path='best_model')
                    logger.info(f"New best FID score: {best_fid}. Model saved.")

            if (epoch + 1) % save_interval == 0:
                save_models(generator, discriminator, optimizer_g, optimizer_d, epoch + 1)

        # Guardar el modelo final
        save_models(generator, discriminator, optimizer_g, optimizer_d, num_epochs, path='final_model')
        end_time = time.time()
        logger.info(f'Training completed in {end_time - start_time:.2f} seconds.')
    except Exception as e:
        logger.error(f'Error during training: {e}', exc_info=True)

if __name__ == "__main__":
    try:
        logger.info("Initializing models and data loader...")
        generator = Generator().to(device)
        generator.apply(weights_init)
        discriminator = Discriminator().to(device)
        discriminator.apply(weights_init)
        dataloader = get_data_loader()
        
        logger.info("Starting training...")
        train(generator, discriminator, dataloader)
    except Exception as e:
        logger.error(f'Error during initialization or training: {e}', exc_info=True)

    logger.info('Training process finished.')