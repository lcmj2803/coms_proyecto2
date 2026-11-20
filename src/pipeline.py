"""Integración: ¿hay señal? -> SNR -> modulación -> confianza."""

from collections import namedtuple

import numpy as np
import torch

from src.detection import (
    N_SAMPLES,
    SNR_PISO_DB,
    detectar_senal,
    estimar_snr_espectral,
)

DESCONOCIDA = "desconocida"

from src.pre_procesamiento import CLASSES, normalize_iq_batch

Resultado = namedtuple(
    "Resultado",
    ["hay_senal", "snr_db", "clase", "confianza"],
)


def iq_a_complejo(obs):
    """Convierte una observación (2, N_SAMPLES) real en (N_SAMPLES,) compleja.

    Fila 0 = I (parte real), fila 1 = Q (parte imaginaria).
    """
    obs = np.asarray(obs)
    if obs.shape != (2, N_SAMPLES):
        raise ValueError(
            "Se esperaba forma (2, {}), llegó {}.".format(N_SAMPLES, obs.shape)
        )
    return obs[0] + 1j * obs[1]


def medir_snr(complejo):
    """Es/N0 espectral en dB, o None si el estimador quedó en el piso.

    SNR_PISO_DB (-20) es el recorte cuando T <= 1, o cuando la cuenta
    cae por debajo. No es una medición de -20 dB.
    """
    snr_db = float(estimar_snr_espectral(complejo))
    if snr_db <= SNR_PISO_DB:
        return None
    return snr_db


def analizar(obs, cnn, umbral=None):
    """Encadena detección, SNR y la CNN.

    obs tiene forma (2, N_SAMPLES). cnn es la red ya cargada.

    hay_senal queda en True solo si el detector ve energía y la CNN
    elige una modulación. Si la red elige "ruido", hay_senal pasa a
    False y la confianza es la de esa clase. Si el detector no ve
    señal, no se llama a la red: clase "ruido" y confianza None.

    umbral (opcional, entre 0 y 1): si la red elige una modulación con
    confianza menor que umbral, la clase pasa a "desconocida". hay_senal
    sigue en True (el detector sí vio energía) y la confianza se
    conserva. Con umbral=None no se rechaza nada.

    snr_db es None bajo el piso del estimador. Al medir el error contra
    snr_inst_db hay que omitir esas filas y las de ruido: en meta.csv
    su referencia ya es nan.
    """
    complejo = iq_a_complejo(obs)
    hay_senal = bool(detectar_senal(complejo))
    snr_db = medir_snr(complejo)

    if not hay_senal:
        return Resultado(False, snr_db, "ruido", None)

    lote = normalize_iq_batch(np.asarray(obs)[None, ...])
    tensor = torch.from_numpy(np.ascontiguousarray(lote))

    cnn.eval()
    with torch.no_grad():
        proba = torch.softmax(cnn(tensor), dim=1)[0]

    indice = int(torch.argmax(proba))
    clase = CLASSES[indice]
    confianza = float(proba[indice])

    if clase == "ruido":
        hay_senal = False
    elif umbral is not None and confianza < umbral:
        clase = DESCONOCIDA

    return Resultado(hay_senal, snr_db, clase, confianza)


