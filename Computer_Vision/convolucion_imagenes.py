import numpy as np

# Definir la matriz de la imagen (4x3)
imagen = np.array([
    [1, 3, 1],
    [3, 1, 5],
    [2, 2, 3],
    [3, 2, 1]
])

# Definir el filtro (2x1)
filtro = np.array([
    [0.5, 0.5],
    [1, -1]
])

# Tamaños de la imagen y el filtro
i_h, i_w = imagen.shape
f_h, f_w = filtro.shape

# Inicializar el resultado
resultado = np.zeros((i_h - f_h + 1, i_w - f_w + 1))

# Realizar la convolución
for y in range(resultado.shape[0]):
    for x in range(resultado.shape[1]):
        subimagen = imagen[y:y+f_h, x:x+f_w]
        resultado[y, x] = np.sum(subimagen * filtro)

print("Resultado de la convolución:")
print(resultado)
