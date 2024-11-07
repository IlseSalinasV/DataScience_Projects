import numpy as np
def convolve(sequence, weights):
    convolution = np.zeros(len(sequence) - len(weights) + 1)
    for i in range(convolution.shape[0]):
        convolution[i] = np.sum(
            np.array(weights) * np.array(sequence[i : i + len(weights)])
        )
    return convolution

s = [2, 3, 5, 7, 11]
w = [-1, 1]

respuesta = convolve(s,w)
print(respuesta)