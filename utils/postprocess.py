import numpy as np
from einops import rearrange, repeat


def compute_stress(inputs, outputs, label):
    """
    inputs: (b, 1, h, w)
    outputs: (b, 9, h, w)
    label: (b, 4)
    """
    # Rename the variables
    b, c, h, w = inputs.shape  # Size of 4D matrix

    inputs = rearrange(inputs, "b c h w -> (h w) (b c)")
    outputs = rearrange(outputs, "b c h w -> (h w) c b")

    # Strain components
    stran1 = outputs[:, 0:3, :].transpose(1, 0, 2).reshape(3, 1, h * w, b)
    stran2 = outputs[:, 3:6, :].transpose(1, 0, 2).reshape(3, 1, h * w, b)
    stran3 = outputs[:, 6:9, :].transpose(1, 0, 2).reshape(3, 1, h * w, b)

    # Compute the material property of each pixel in each RVE
    cons = np.zeros((3, 3, h * w, b), dtype=np.float32)
    for i in range(b):
        matrix = np.where(inputs[:, i] == 0)[0]
        fiber = np.where(inputs[:, i] == 1)[0]
        E0, v0, E1, v1 = label[i]

        A = (
            E0
            / ((1 + v0) * (1 - 2 * v0))
            * np.array([[1 - v0, v0, 0], [v0, 1 - v0, 0], [0, 0, (1 - 2 * v0) / 2]])
        )
        B = (
            E1
            / ((1 + v1) * (1 - 2 * v1))
            * np.array([[1 - v1, v1, 0], [v1, 1 - v1, 0], [0, 0, (1 - 2 * v1) / 2]])
        )

        cons[:, :, matrix, i] = repeat(A, "i j -> i j m", m=len(matrix))
        cons[:, :, fiber, i] = repeat(B, "i j -> i j f", f=len(fiber))

    # Compute stress
    stress1 = np.einsum("ijkm,jlkm->ilkm", cons, stran1)
    stress2 = np.einsum("ijkm,jlkm->ilkm", cons, stran2)
    stress3 = np.einsum("ijkm,jlkm->ilkm", cons, stran3)

    # Homogenization - average stress
    s1avg = np.sum(stress1, axis=2).reshape(3, b) / (h * w)
    s2avg = np.sum(stress2, axis=2).reshape(3, b) / (h * w)
    s3avg = np.sum(stress3, axis=2).reshape(3, b) / (h * w)

    # # Get the diagonal and off-diagonal parts
    Cdiag = 0.5 * (s1avg[0, :] + s2avg[1, :])
    C_offdiag = 0.5 * (s1avg[1, :] + s2avg[0, :])
    C33 = s3avg[2, :]

    # # Compute the average Young's modulus and Poisson's ratio
    v_avg = C_offdiag / (Cdiag + C_offdiag)
    E_avg = 2 * C33 * (1 + v_avg)

    return v_avg, E_avg
