from PIL import Image
import os

def resize_and_organize_images(input_dir, output_dir, size=(128, 128)):
    print("Iniciando procesamiento de imágenes...")

    for root, subdirs, files in os.walk(input_dir):
        for subdir in subdirs:
            # Crear el directorio de salida para cada clase
            output_class_dir = os.path.join(output_dir, subdir)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)

            class_input_dir = os.path.join(root, subdir)
            for filename in os.listdir(class_input_dir):
                print("Procesando:", filename)
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(class_input_dir, filename)
                    img = Image.open(img_path)
                    img = img.resize(size, Image.LANCZOS)
                    
                    # Guardar en el directorio de salida por clase
                    img.save(os.path.join(output_class_dir, filename))

resize_and_organize_images('./images/archive/lfw-deepfunneled/lfw-deepfunneled', './images/output')

