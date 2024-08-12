import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import logging
from tqdm import tqdm
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Definir el generador (asegúrate de que esta definición coincida con la del script de entrenamiento)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def load_generator(model_path):
    logger.info(f"Loading generator from {model_path}")
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    logger.info("Generator loaded successfully")
    return generator

def generate_images(generator, num_images, output_dir):
    logger.info(f"Generating {num_images} images")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(num_images), desc="Generating images"):
            noise = torch.randn(1, 100, 1, 1, device=device)
            fake_image = generator(noise)
            save_image(fake_image, f"{output_dir}/generated_image_{i+1}.png", normalize=True)
    
    end_time = time.time()
    logger.info(f"Image generation completed in {end_time - start_time:.2f} seconds")

def main():
    # Configuración
    model_path = 'final_model/gan_checkpoint_epoch_10.pth'  # Ajusta esto a la ubicación de tu modelo guardado
    output_dir = 'generated_test_images'
    num_images = 10  # Número de imágenes a generar

    try:
        # Cargar el generador
        generator = load_generator(model_path)

        # Generar imágenes
        generate_images(generator, num_images, output_dir)

        logger.info(f"All {num_images} images have been generated and saved in {output_dir}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()