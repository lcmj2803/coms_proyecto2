import numpy as np


CLASSES = (
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "16APSK",
    "ruido",
)

CLASS_TO_INDEX = {
    name:index 
    for index, name in enumerate(CLASSES)
}


def normalize_iq_batch(batch):
    # Copiar para no modificar las señales originales.
    x = np.array(batch, dtype=np.float32, copy=True)

    if x.ndim != 3 or x.shape[1:] != (2, 1024):
        raise ValueError("Se esperaba un lote con forma (B, 2, 1024).")

    if not np.all(np.isfinite(x)):
        raise ValueError("El lote contiene NaN o valores infinitos.")

    # Quitar por separado la media temporal de I y de Q.
    x -= np.mean(x, axis=2, keepdims=True)

    # |x[n]|² = I[n]² + Q[n]².
    # Obtener una potencia por observación.
    power = np.mean(
        x[:, 0, :] ** 2 + x[:, 1, :] ** 2,
        axis=1
    )

    if np.any(power <= 0) or not np.all(np.isfinite(power)):
        raise ValueError("Hay observaciones con potencia nula o no válida.")

    # Un mismo divisor para ambos canales y todas las muestras.
    x /= np.sqrt(power)[:, None, None]

    return x