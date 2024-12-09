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



## Parámetros del entrenamiento

### Las clases de nuestra base de datos

```Python
classes = ("Clase 1", "Clase 2", "Clase n")
```

### Definimos el tamaño del *batch*, es decir, el tamaño del conjunto de muestras para calcular la dirección del gradiente, pues usaremos la estrategia *mini-batch SGD*

```Python
batchSize = 4
```


## Cargar imágenes y conjuntos

### Definimos las transformaciones que le vamos a aplicar a las imágenes de la base de datos

- Opción 1:

    ```Python
    transform = transforms.Compose([

        # Serie de transformaciones
        transformacion_1(),
        transformacion_1(),
        transformacion_n(),

        # Transformación a tensor
        ToTensor(),

        # Normalización
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

- Opción 2:

    ```Python
    transform = transforms.Compose([

        transformacion_1(),
        transformacion_1(),
        transformacion_n(),

        # Transformación a tensor
        transforms.ToTensor(),

        # Normalización
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    ```

### Definir los conjuntos de entrenamiento y testeo para una base de datos genérica

```Python
# Definimos el conjunto de entrenamiento a partir de una base de datos estándar (CIFAR10):
# - Se almacenará en el directorio local "data" (root="./data").
# - Lo descargaremos localmente (download=True).
# - Aplicaremos las transformaciones que hemos decidido (transform=transform).
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Definimos el conjunto de testeo a partir de una base de datos estándar (CIFAR10):
# - Se almacenará en el directorio local "data" (root="./data").
# - Lo descargaremos localmente (download=True).
# - Aplicaremos las transformaciones que hemos decidido (transform=transform).
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
```

### Uso de la clase *Dataset*

La clase `torch.utils.data.Dataset` es una clase abstracta que representa un conjunto de datos cargado de una base de datos local. Para crear nuestro propio conjunto de datos en *pytorch* debemos heredar de dicha clase y sobreescribir los siguientes métodos:

- `__init__`: Es el método constructor, encargado de leer e indexar la base de datos. Tiene los siguientes argumentos:

    - `csv_file` (*string*): Ruta al fichero CSV con las anotaciones.

    - `root_dir` (*string*): Directorio raíz donde encontraremos las carpetas "images" y "masks".

    - `transform` (*callable*, *optional*): Transformaciones opcionales a realizar sobre las imágenes.

    - `maxSize` (un número): Número máximo de imágenes a incluir en la base de datos, útil para ejecutar más rápido sobre un subconjunto de los datos. Si `maxSize=0`, se usan todos los datos.

    - `classes` (lista): Lista con las clases de la base de datos.

- `__len__`: Es el método que permite invocar `len(dataset)`, que nos devuelve el tamaño del conjunto de datos.

- `__getitem__`: Es para soportar el indexado `dataset[i]` al referirnos a la muestra `i`.

    Con *Dataset*, podemos crear los conjuntos de datos de nuestro problema de diagnóstico. Podemos leer el fichero CSV en el método de inicialización `__init__` pero dejar la lectura explícita de las imágenes para el método `__getitem__`. Esta aproximación es más eficiente en memoria porque todas las imágenes no se cargan en memoria al principio, sino que se van leyendo individualmente cuando es necesario.

    Cada muestra de nuestro conjunto de datos (cuando invoquemos `dataset[i]`) va a ser un diccionario:

    ```Python
    {
        "image": image,
        "mask": mask,
        "label": label
    }
    ```

    Por otro lado, al definir el conjunto de datos, el constructor podrá también tomar un argumento opcional `transform` para que podamos añadir pre-procesado y técnicas de *data augmentation* que le aplicaremos a las imágenes cuando las solicitemos.

La clase es la siguiente:

```Python
class MyDataset(Dataset):

    # Sobreescribimos el método "__init__".
    def __init__(self, csv_file, root_dir, transform=None, maxSize=0, classes):

        # Definimos el conjunto de datos a partir del fichero CSV
        self.dataset = pd.read_csv(csv_file, header=0, dtype={"id": str, "label": int})

        # Se realizan las operaciones necesarias en el caso de que se haya definido un número máximo de imágenes.
        if maxSize>0:

            newDatasetSize = maxSize

            # El nuevo conjunto de datos será una parte del antiguo seleccionada de manera aleatoria
            idx = np.random.RandomState(seed=42).permutation(range(len(self.dataset)))
            print(idx[0:newDatasetSize])
            reduced_dataset = self.dataset.iloc[idx[0:newDatasetSize]]

            # Se reajusta el conjunto de datos al límite impuesto
            self.dataset = reduced_dataset.reset_index(drop=True)

        # Se definen los directorios locales con los datos a trabajar
        self.root_dir = root_dir # Directorio raíz
        self.img_dir = os.path.join(root_dir, "images") # Directorio de imágenes
        self.mask_dir = os.path.join(root_dir, "masks") # Directorio de máscaras (no todos los problemas lo requieren)

        # Se definen las transformaciones
        self.transform = transform

        # Se definen las clases de la base de datos
        self.classes = classes

    # Sobreescribimos el método "__len__".
    def __len__(self):

        return len(self.dataset)

    # Sobreescribimos el método "__getitem__".
    def __getitem__(self, idx):

        # Si es un tensor, se convierte a lista.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Leemos la imagen.
        img_name = os.path.join(self.img_dir, self.dataset.id[idx] + ".jpg")
        image = io.imread(img_name)

        # Leemos la máscara.
        mask_name = os.path.join(self.mask_dir, self.dataset.id[idx] + ".png")
        mask = io.imread(mask_name)

        # La muestra será un triplete imagen-máscara-etiqueta.
        sample = {
            "image": image,
            "mask": mask,
            "label":  self.dataset.label[idx].astype(dtype=np.long)
        }

        # Se transforma la imagen si es necesario.
        if self.transform:
            sample = self.transform(sample)

        return sample
```

### Para cargar un conjunto de datos con la clase *Dataset*

```Python
# Cargamos el conjunto da datos de entrenamiento
trainset = MyDataset(
    csv_file="data/DBtrain.csv",
    root_dir="data",
    transform=transform,
    maxSize=500
)

# Cargamos el conjunto da datos de testeo
testset = MyDataset(
    csv_file="data/DBtest.csv",
    root_dir="data",
    transform=transform,
    maxSize=500
)

# Cargamos el conjunto da datos de validación
valset = MyDataset(
    csv_file="data/DBval.csv",
    root_dir="data",
    transform=transform,
    maxSize=500
)
```

### Funciones de carga de las bases de datos

```Python
# Definimos la función de carga de datos para el entrenamiento:
# - Le asignamos el set de entrenamiento.
# - Definimos batches del tamaño indicado (batch_size=batchSize).
# - Desordenamos las imágenes (shuffle=True).
# - Usamos 2 cores para paralelizar la carga de datos y agilizar el proceso de lectura (num_workers=2).
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

# Definimos la función de carga de datos para el entrenamiento
# - Le asignamos el set de entrenamiento.
# - Definimos batches del tamaño indicado (batch_size=batchSize).
# - En test no desordenamos (shuffle=False).
# - Usamos 2 cores para paralelizar la carga de datos y agilizar el proceso de lectura (num_workers=2).
testloader = DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

# Definimos la función de carga de datos para la validación
# - Le asignamos el set de validación.
# - Definimos batches del tamaño indicado (batch_size=batchSize).
# - En test no desordenamos (shuffle=False).
# - Usamos 2 cores para paralelizar la carga de datos y agilizar el proceso de lectura (num_workers=2).
testloader = DataLoader(valset, batch_size=batchSize, shuffle=False, num_workers=2)
```



## Funciones y acciones útiles

### Si usamos una base de datos local con estructura de fichero CSV, es de utilidad crear una variable que nos permite indexar las imágenes

```Python
# Leemos la base de datos
db = pd.read_csv(
    "<ruta al fichero CSV>",
    header=0,
    dtype={
        "Campo de cadena": str,
        "Campo numérico": int
    }
)
```

### Definimos una función que nos permite visualizar una imagen a partir de un tensor torch normalizado

```Python
def imshow_norm(img):

    # Desnormalizamos el tensor torch
    img = img/2 + 0.5

    # Convertimos el tensor en una matriz numpy
    npimg = img.numpy()

    # Reordenamos dimensiones: el tensor torch es 3xHxW y pasa a HxWx3
    plt.imshow(np.transpose(
        npimg,
        (1, 2, 0)
    ))

    # Lo mostramos
    plt.show()
```

### Definimos una función que nos permite visualizar una imagen con un título

```Python
def imshow_prov(image, title_str):

    if len(image.shape) > 2:
        plt.imshow(image)

    else:
        plt.imshow(image, cmap=plt.cm.gray)

    plt.title(title_str)
```

### Mostrar un batch de imágenes de entrenamiento

```Python
# Generamos un iterador sobre el cargador de imágenes
dataiter = iter(trainloader)

# Obtenemos las siguientes imágenes y sus etiquetas
images, labels = next(dataiter)

# El tensor "images" tiene dimensiones NxCxHxW, es decir:
# - N imágenes en el batch
# - C canales
# - H,W dimensiones espaciales
print("Tamaño de la imagen en NxCxHxW: " + str(images.size()))

# Las mostramos (make_grid las concatena en un grid espacial para mostrarlas todas juntas)
imshow_norm(torchvision.utils.make_grid(images))

# Mostramos las etiquetas
print(" ".join("%5s" % classes[labels[j]] for j in range(batchSize)))
```

### Mostrar un batch de imágenes de test

```Python
# Iteramos el test
dataiter = iter(testloader)

# Obtenemos imágenes y las etiquetas
images, labels = next(dataiter)

# El tensor "images" tiene dimensiones NxCxHxW, es decir:
# - N imágenes en el batch
# - C canales
# - H,W dimensiones espaciales
print("Tamaño de la imagen en NxCxHxW: " + str(images.size()))

# Las mostramos (make_grid las concatena en un grid espacial para mostrarlas todas juntas)
imshow_norm(torchvision.utils.make_grid(images))

# Mostramos las etiquetas
print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(batchSize)))
```

### Mostrar una imagen local

```Python
# Nueva figura
plt.figure()

# Mostramos la imagen
imshow_prov(io.imread(os.path.join("<CARPETA DE IMÁGENES>/", img_id + ".jpg" )), "Imagen %d"%n)

# Cargamos la gráfica
plt.show()
```

### Mostramos los datos de la imagen indicada en una base de datos local

```Python
# Índice de la imagen
n = 65

# Obtenemos el ID y la etiqueta
img_id = db.id[n]
label = db.label[n]

# Mostramos la información
print("Mostrando datos de la imagen {}:".format(n))
print("\t- Image ID: {}".format(img_id))
print("\t- Label: {}".format(label))
```

### Mostrar imágenes de un conjunto cargado a la clase *Dataset*

```Python
# Creamos una figura
fig = plt.figure()

# Recorremos el conjunto de datos para mostrar el número indicado de muestras
N = 4;
for i in range(N):

    # Accedemos a una muestra
    sample = trainset[i]
    print(i, sample["image"].shape, sample["label"])

    # Mostramos la muestra
    ax = plt.subplot(1, N, i + 1)
    plt.tight_layout()
    ax.set_title("Sample #{}".format(i))
    ax.axis("off")
    plt.imshow(sample["image"])

    # Nos detenemos si alcanzamos la longitud de la base de datos
    if i == len(trainset):
        break

# Mostramos la gráfica
plt.show()
```

### Definimos una función auxiliar para visualizar un batch de datos

```Python
def show_batch(sample_batched):

    # Mostramos un batch
    images_batch, labels_batch = sample_batched["image"], sample_batched["label"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    # Generamos el grid
    grid = utils.make_grid(images_batch)

    # Lo pasamos a numpy y lo desnormalizamos
    grid = grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)

    # Mostramos el batch
    plt.imshow(grid)
    plt.title("Batch from dataloader")
```

### Mostramos varios batches del conjunto de datos de entrenamiento

```Python
# Cuántos batches vamos a mostrar
N = 4

# Iteramos en el conjunto de datos de entrenamiento
for i, sample_batched in enumerate(trainloader):

    print(i, sample_batched["image"].size(), sample_batched["label"])

    # Mostramos cada batch en una sola figura
    plt.figure()
    show_batch(sample_batched)
    plt.axis("off")
    plt.ioff()
    plt.show()

    # Mostramos datos de los 4 primeros batches y paramos.
    if i == N-1:
        break
```