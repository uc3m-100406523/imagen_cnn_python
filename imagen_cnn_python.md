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