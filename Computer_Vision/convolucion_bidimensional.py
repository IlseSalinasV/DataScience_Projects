import numpy as np
def convolve2d(S, W):
    # Tamaños de las matrices
    n, m = S.shape
    p, q = W.shape

    # Tamaño de la salida
    output_height = n - p + 1
    output_width = m - q + 1

    # Inicializar la matriz de salida
    C = np.zeros((output_height, output_width))

    # Realizar la convolución bidimensional
    for i in range(output_height):
        for j in range(output_width):
            C[i, j] = np.sum(S[i:i + p, j:j + q] * W)

    return C

S = np.array([
    [2, 1, 3],
    [4, 0, 2],
    [1, 5, 6]
])

W = np.array([
    [-1, -2],
    [1,   2]
])

respuesta = convolve2d(S,W)
print(respuesta)