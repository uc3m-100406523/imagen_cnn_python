Análisis de imagen con CNN (Python)
===================================



## Bibliotecas

### Interacción con el entorno
```Python
# Biblioteca para trabajar en Google Drive
from google.colab import drive

# Bibliotecas para trabajar con el sistema de ficheros
import os, sys
from shutil import copyfile
```

### Bibliotecas para el uso de GPU (principales)

```Python
# Biblioteca de...
import torch

# Biblioteca de...
import torchvision
```

### Bibliotecas para el uso de GPU (anejas)

```Python
# Bibliotecas para cargar y transformar imágenes
from torchvision import transforms, utils, models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Bibliotecas de redes neuronales
import torch.nn as nn
import torch.nn.functional as F

# Biblioteca de optimización y función de pérdida
import torch.optim as optim
from torch.optim import lr_scheduler
```

### Bibliotecas de matemáticas

```Python
# Bibliotecas de álgebra
import numpy as np

# Bibliotecas para generar números aleatorios
import random
import numpy.random as npr

import numbers
```

### Biblioteca para testear la red

```Python
from sklearn import metrics
```

### Bibliotecas de gráficas

```Python
import matplotlib.pyplot as plt
```

### Otras bibliotecas

```Python
# Biblioteca para calcular tiempos
import time

# Biblioteca para manejar ficheros ZIP
import zipfile

# Biblioteca para el manejo de datos
import pandas as pd

# Biblioteca de advertencias
import warnings

# Biblioteca de...
from __future__ import print_function, division

# Biblioteca de...
from skimage import io, transform, util

# Biblioteca de...
import copy

# Biblioteca de...
import pdb

# Biblioteca de...
from PIL import Image
```



## Uso de cuadernos en *Google Colab*

Mostrar las gráficas en el cuaderno

```Python
%matplotlib inline
```

Usar el sistemas de ficheros de Google Drive

```Python
# Se monta la unidad
drive.mount("/content/drive")

# Se muestra el directorio de trabajo actual
print(os.getcwd())

# Accedemos al directorio de trabajo
os.chdir("<ruta al directorio>")

# Copiar un fichero
copyfile("<ruta al fichero de origen>", "<ruta al fichero de destino>")
```



## Uso de GPU

### Modo interactivo

```Python
plt.ion()
```

### Se selecciona la GPU

```Python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```



## Utilidades varias

### Generación de números aleatorios

```Python
random.seed(42)
npr.seed(42)
torch.manual_seed(42)
```

### Ignora las advertencias

```Python
warnings.filterwarnings("ignore")
```

### Extraer fichero

```Python
with zipfile.ZipFile("<RUTA>", "r") as zip_ref:
    zip_ref.extractall("<DIRECTORIO DE SALIDA>")
```

### Leer fichero CSV

```Python
db = pd.read_csv(
    "<RUTA>",
    header=0,
    dtype={
        "Campo de cadena": str,
        "Campo numérico": int
    }
)
```



## Preprocesado de datos

### Recortamos la imagen usando la máscara de la lesión

```Python
class CropByMask(object):

    # Método de inicialización. Argumentos:
    #   border (tupla ó int): El borde de recorte alrededor de la máscara. Es sabido que el análisis del borde de la lesión con la piel circudante es importante para los dermatólogos, por lo que puede ser interesante dejar una guarda. Si es una tupla, entonces es (bordery, borderx).
    def __init__(self, border):

        # Para comprobar el formato de la información
        assert isinstance(border, (int, tuple))

        # Si es int
        if isinstance(border, int):
            self.border = (border, border)

        # Si es una tupla
        else:
            self.border = border

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Se obtienen las dimensiones de la imagen
        h, w = image.shape[:2]

        # Calculamos los índices del "bounding box" para hacer el "cropping"
        sidx = np.nonzero(mask)
        minx = np.maximum(sidx[1].min()-self.border[1], 0)
        maxx = np.minimum(sidx[1].max()+1+self.border[1], w)
        miny = np.maximum(sidx[0].min()-self.border[0], 0)
        maxy = np.minimum(sidx[0].max()+1+self.border[1], h)

        # Recortamos la imagen
        image = image[miny:maxy, minx:maxx, ...]
        mask = mask[miny:maxy, minx:maxx]

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Recortamos aleatoriamente la imagen

```Python
class RandomCrop(object):

    # Método de inicialización. Argumentos:
    #   output_size (tupla ó int): Tamaño del recorte. Si int, recorte cuadrado.
    def __init__(self, output_size):

        # Para comprobar el formato de la información
        assert isinstance(output_size, (int, tuple))

        # Si es int
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        # Si es una tupla
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Se obtienen las dimensiones de la imagen
        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        if h>new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top=0

        if w>new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        image = image[top: top + new_h, left: left + new_w]
        mask = mask[top: top + new_h, left: left + new_w]

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Recortamos el área central de la imagen

```Python
class CenterCrop(object):

    # Método de inicialización. Argumentos:
    #   output_size (tupla or int): El tamaño deseado. Si es int, el recorte es cuadrado.
    def __init__(self, output_size):

        # Para comprobar el formato de la información
        assert isinstance(output_size, (int, tuple))

        # Si es int
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        # Si es una tupla
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Se obtienen las dimensiones de la imagen
        h, w = image.shape[:2]

        new_h, new_w = self.output_size
        rem_h = h - new_h
        rem_w = w - new_w

        if h>new_h:
            top = int(rem_h/2)
        else:
            top=0

        if w>new_w:
            left = int(rem_w/2)
        else:
            left = 0

        image = image[top: top + new_h, left: left + new_w]

        mask = mask[top: top + new_h, left: left + new_w]

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Rescalamos la imagen a un tamaño determinado

```Python
class Rescale(object):

    # Método de inicialización. Argumentos:
    #   output_size (tupla ó int): El tamaño deseado. Si es una tupla, "output" es el "output_size". Si es un int, la dimensión más pequeña será el "output_size" y mantendremos la relación de aspecto original.
    def __init__(self, output_size):

        # Para comprobar el formato de la información
        assert isinstance(output_size, (int, tuple))

        # Se actualiza el tamaño de salida
        self.output_size = output_size

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Se obtienen las dimensiones de la imagen
        h, w = image.shape[:2]

        # Si es int
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        # Si es una tupla
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        msk = transform.resize(mask, (new_h, new_w))

        # Se devuelve la muestra
        return {
            "image": img,
            "mask": msk,
            "label" : label
        }
```

### Recortamos el área central de la imagen

```Python
class TVCenterCrop(object):

    def __init__(self, size):

        # Se importa la transformación desde la librería torchvision.transforms
        self.CC = transforms.CenterCrop(size)

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.CC(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.CC(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Recortamos el área central de la imagen con un tamaño aleatorio

```Python
class TVRandomCenterCrop(object):

    def __init__(self):

        # Tamaño aleatorio entre 100 y 224
        size = random.randint(100, 224)

        # Se importa la transformación desde la librería torchvision.transforms
        self.RC = TVCenterCrop(size)

    def __call__(self, sample):

        # Se devuelve la muestra
        return self.RC(sample)
```

### Recortamos la imagen usando la máscara de la lesión

```Python
class TVCropByMask(object):

    def __init__(self, border):

        # Para comprobar el formato de la información
        assert isinstance(border, (int, tuple))

        # Si es int
        if isinstance(border, int):
            self.border = (border, border)

        # Si es una tupla
        else:
            self.border = border

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Se obtienen las dimensiones de la imagen
        h, w = image.shape[:2]

        # Calculamos los índices del "bounding box" para hacer el "cropping"
        sidx = np.nonzero(mask)
        minx = np.maximum(sidx[1].min()-self.border[1], 0)
        maxx = np.minimum(sidx[1].max()+1+self.border[1], w)
        miny = np.maximum(sidx[0].min()-self.border[0], 0)
        maxy = np.minimum(sidx[0].max()+1+self.border[1], h)

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = TF.crop(pil_image, top=miny, left=minx, height=maxy-miny, width=maxx-minx)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = TF.crop(pil_mask, top=miny, left=minx, height=maxy-miny, width=maxx-minx)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Recortamos aleatoriamente la imagen

```Python
class TVRandomCrop(object):

    def __init__(self, size):

        # Se importa la transformación desde la librería torchvision.transforms
        self.RC = transforms.RandomCrop(size)

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.RC(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.RC(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Re-escalamos la imagen a un tamaño determinado

```Python
class TVRescale(object):

    def __init__(self, size):

        # Se importa la transformación desde la librería torchvision.transforms
        self.Res = transforms.Resize(size)

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.Res(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.Res(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Rotación aleatoria

```Python
class TVRandomRotation(object):

    def __init__(self):

        # Se importa la transformación desde la librería torchvision.transforms
        self.RR = transforms.RandomRotation(180, expand=True)

        # Se importa la transformación de recorte con máscara
        self.CBM = TVCropByMask(0)

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.RR(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.RR(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Giro horizontal aleatorio

```Python
class TVRandomHorizontalFlip(object):

    def __init__(self):

        # Se importa la transformación desde la librería torchvision.transforms
        self.RHF = transforms.RandomHorizontalFlip()

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.RHF(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.RHF(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Giro vertical aleatorio

```Python
class TVRandomVerticalFlip(object):

    def __init__(self):

        # Se importa la transformación desde la librería torchvision.transforms
        self.RVF = transforms.RandomVerticalFlip()

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # La imagen se convierte a tipo PILImage
        pil_image = Image.fromarray(util.img_as_ubyte(image))

        # Transformación de la imagen
        pil_image = self.RVF(pil_image)

        # Se convierte de vuelta la imagen PILImage a scikit-image
        image = util.img_as_float(np.asarray(pil_image))

        # La máscara se convierte a tipo PILImage
        pil_mask = Image.fromarray(util.img_as_ubyte(mask))

        # Transformación de la máscara
        pil_mask = self.RVF(pil_mask)

        # Se convierte de vuelta la máscara PILImage a scikit-image
        mask = util.img_as_float(np.asarray(pil_mask))

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Cambiar color de piel

```Python
class ChangeSkin(object):

    def __init__(self, colour):

        # Para comprobar el formato de la información
        assert isinstance(colour, (int, tuple))

        # Si es int
        if isinstance(colour, int):
            num_list = [colour%256, colour%256, colour%256]
            self.colour = np.array(num_list)

        # Si es una tupla
        else:
            num_list = list(colour)
            for i, elem in enumerate(num_list):
                num_list[i] = elem%256
            self.colour = np.array(num_list)

    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Recorremos la máscara
        for i, row in enumerate(mask):
            for j, elem in enumerate(row):
                if elem == 0:
                    image[i, j] = self.colour

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Cambiar color de piel de manera aleatoria

```Python
class RandomChangeSkin(object):

    def __init__(self):

        # Lista de tonos de piel humana
        self.skin_tones = [
            (0, 0, 0), # Dejar la piel como está
            (141, 85, 36),
            (198, 134, 66),
            (224, 172, 105),
            (241, 194, 125),
            (255, 219, 172),
            (143, 91, 72),
            (128, 81, 64),
            (114, 72, 55),
            (99, 62, 47),
            (84, 53, 38),
            (255, 224, 189),
            (255, 205, 148),
            (234, 192, 134),
            (255, 173, 96),
            (255, 227, 159),
            (244, 204, 199),
            (247, 213, 208),
            (255, 230, 226),
            (255, 238, 231),
            (255, 231, 223),
            (250, 219, 208),
            (235, 204, 171),
            (210, 153, 108),
            (195, 124, 77),
            (182, 107, 62),
            (142, 75, 50),
            (244, 194, 194),
            (250, 235, 230),
            (244, 217, 210),
            (244, 208, 205),
            (219, 173, 146),
            (230, 192, 172),
            (239, 212, 200),
            (235, 201, 197),
            (227, 186, 186),
            (204, 154, 139),
            (212, 169, 156),
            (220, 183, 173),
            (227, 198, 190),
            (235, 212, 207),
            (218, 176, 130),
            (208, 161, 115),
            (190, 142, 96),
            (212, 166, 127),
            (218, 176, 148),
            (195, 150, 123),
            (208, 162, 134),
            (235, 199, 175),
            (226, 182, 162),
            (218, 172, 147),
            (254, 182, 183),
            (252, 202, 191),
            (255, 221, 217),
            (255, 230, 222),
            (38, 7, 1),
            (61, 12, 2),
            (132, 55, 34),
            (175, 110, 81),
            (198, 144, 118),
            (74, 51, 45),
            (198, 141, 130),
            (237, 216, 199)
        ]

    def __call__(self, sample):

        # Se escoge un tono de piel aleatorio
        tone = random.choice(self.skin_tones)

        # Transformamos la muestra
        if tone != (0, 0, 0):
            CS = ChangeSkin(tone)
            sample = CS(sample)

        # Se devuelve la muestra
        return sample
```

### Normalizamos los datos restando la media y dividiendo por las desviaciones típicas

```Python
class Normalize(object):

    # Método de inicialización. Argumentos:
    #   mean_vec: El vector con las medias.
    #   std_vec: el vector con las desviaciones típicas.
    def __init__(self, mean, std):

        assert len(mean)==len(std),"Length of mean and std vectors is not the same"

        self.mean = np.array(mean)
        self.std = np.array(std)

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"],sample["label"]

        # Se obtienen las dimensiones de la imagen
        c, h, w = image.shape

        assert c==len(self.mean), "Length of mean and image is not the same"

        dtype = image.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)

        image.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label" : label
        }
```

### Convertimos ndarrays de la muestra en tensores

```Python
class ToTensor(object):

    # Método de llamada. Argumentos:
    #   sample: Imagen que transformar en formato de triplete imagen-máscara-etiqueta.
    def __call__(self, sample):

        # Se accede al triplete imagen-máscara-etiqueta.
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Cambiamos los ejes
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        # A la máscara le añadimos una dim fake al principio
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = mask.unsqueeze(0)
        label=torch.tensor(label,dtype=torch.long)

        # Se devuelve la muestra
        return {
            "image": image,
            "mask": mask,
            "label": label
        }
```

### Convertir las imágenes a un tensor

```Python
transforms.ToTensor()
```

### Normalizar las imágenes restando 0.5 y diviendo por 0.5 cada canal

```Python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

### Transformación compuesta

```Python
composed = transforms.Compose([
    transformacion_1(),
    transformacion_1(),
    transformacion_n()
])
```