import numpy as np

from src.generator import rrc_taps
from src.dataset import SPS, RRC_BETA, RRC_SPAN

# Filtro adaptado: el mismo RRC del transmisor (0.35, 8, 10) -> 10*8 + 1 = 81 taps
H_RRC = rrc_taps(RRC_BETA, SPS, RRC_SPAN)


def a_complejo(obs):
    """Convierte una observación (2, N) [fila 0 = I, fila 1 = Q] en una señal compleja (N,).

    Es el inverso de lo que hace dataset.py: np.stack([y.real, y.imag]).
    """
    obs = np.asarray(obs)
    if obs.shape[0] != 2:
        raise ValueError(f"se esperaba forma (2, N), llegó {obs.shape}")
    return obs[0] + 1j * obs[1]


def filtro_adaptado(y, h=H_RRC):
    """Filtra y con h y devuelve solo la zona donde el filtro está lleno de señal.

    Con un número impar de taps, cada salida corresponde al centro de su ventana.
    """
    if len(y) < len(h):
        return np.empty(0, dtype=np.result_type(np.asarray(y), np.asarray(h)))
    return np.convolve(y, h, mode="valid")

def temporizacion(z):
    """Devuelve los símbolos y el desplazamiento k de mayor potencia media."""
    z = np.asarray(z)

    if z.ndim != 1 or z.size < SPS:
        raise ValueError("z debe ser un vector con al menos un período de símbolo.")

    # Con 944 muestras y SPS=8, la matriz tiene 118 filas y 8 columnas.
    # La columna k contiene z[k], z[k + 8], z[k + 16], ...
    n_simbolos = z.size // SPS
    fases = z[:n_simbolos * SPS].reshape(n_simbolos, SPS)

    potencia_por_k = np.mean(np.abs(fases) ** 2, axis=0)
    k = int(np.argmax(potencia_por_k))

    return fases[:, k], k


def normalizar_rms(simbolos):
    """Lleva el RMS de los símbolos a 1 y conserva el giro de la constelación."""
    simbolos = np.asarray(simbolos)

    if simbolos.ndim != 1 or simbolos.size == 0:
        raise ValueError("Se esperaba un vector de símbolos no vacío.")

    rms = np.sqrt(np.mean(np.abs(simbolos) ** 2))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError("Se necesita un RMS finito y positivo.")

    return simbolos / rms

N_FFT = 8192

# Tolerancia de diseño del receptor: |CFO| <= 3e-3 ciclos/símbolo.
# Cubre train (±1e-4 ciclos/muestra x 8 = ±8e-4) y exp6 (±3e-4 x 8 = ±2.4e-3)
# con margen. No es un oráculo: es el rango que el receptor acepta buscar.
CFO_MAX_SIMBOLO = 3e-3

# Umbral de la fuerza de la línea, calibrado en train (2026-11-25):
# con solo ruido (Es/N0 <= 0 dB) la fuerza queda bajo 0.22 en ~97-99 % de los casos.
# Con 0.22 la mediana del error de CFO nunca supera a "predecir CFO = 0".
UMBRAL_LINEA = 0.22


def estimar_cfo(s, M):
    """Devuelve el CFO en ciclos/símbolo y la fuerza de la línea espectral de s**M.

    Solo busca el pico en |f| <= M * CFO_MAX_SIMBOLO: fuera de ahí cualquier
    pico es ruido o modulación, no CFO.

    fuerza = |pico| / sqrt(N * sum|s**M|^2), entre 0 y 1:
      - tono puro (todas las s**M con la misma fase)  -> 1
      - ruido, o una sola muestra que domina la suma  -> ~1/sqrt(N) ≈ 0.09 con N = 118
    (Con sum|s**M| en el denominador, una muestra dominante daba ~1: el ruido
    parecía línea.)
    """
    if M not in (4, 8):
        raise ValueError("M debe ser 4 u 8.")

    s = np.asarray(s)
    if s.ndim != 1 or not 0 < s.size <= N_FFT:
        raise ValueError("Se requieren entre 1 y {} símbolos.".format(N_FFT))

    potencia_m = s ** M
    energia = np.sum(np.abs(potencia_m) ** 2)
    if not np.isfinite(energia) or energia == 0:
        raise ValueError("La energía de s**M debe ser finita y positiva.")

    espectro = np.abs(np.fft.fft(potencia_m, n=N_FFT))
    frecuencias = np.fft.fftfreq(N_FFT, d=1.0)

    # Índices de los bins dentro de la ventana física; el pico se busca solo ahí.
    # Ejemplo: M = 4 -> |f| <= 0.012 ciclos/símbolo -> ~197 bins de 8192.
    en_ventana = np.flatnonzero(np.abs(frecuencias) <= M * CFO_MAX_SIMBOLO)
    pico = en_ventana[np.argmax(espectro[en_ventana])]

    cfo = frecuencias[pico] / M
    fuerza = np.clip(espectro[pico] / np.sqrt(s.size * energia), 0.0, 1.0)
    return float(cfo), float(fuerza)


def elegir_m(s, umbral=UMBRAL_LINEA):
    """Devuelve (M, cfo, fuerza). Prefiere M = 4; M = 8 solo si M = 4 no tiene línea.

    Por qué no gana simplemente la mayor fuerza: en QAM, M = 8 también da el CFO
    correcto (8 es múltiplo de 4), pero deja la fase conocida solo módulo 45°,
    y la mitad de las observaciones terminaba girada 45° (rejilla en rombo).

    M = 0 significa "no hay línea": cfo = 0 y no se corrige la fase.
    Pasa en 16APSK (su anillo de 12 puntos se cancela en s**4 y s**8) y a SNR bajo.
    """
    cfo4, fuerza4 = estimar_cfo(s, 4)
    if fuerza4 >= umbral:
        return 4, cfo4, fuerza4

    cfo8, fuerza8 = estimar_cfo(s, 8)
    if fuerza8 >= umbral:
        return 8, cfo8, fuerza8

    return 0, 0.0, max(fuerza4, fuerza8)


def simbolos_corregidos(obs):
    """Recupera 118 símbolos complejos de una observación I/Q de 1024 muestras."""
    z = filtro_adaptado(a_complejo(obs))
    s, _ = temporizacion(z)
    s = normalizar_rms(s)

    M, cfo, _ = elegir_m(s)
    if M == 0:
        # Sin línea no hay estimación confiable: mejor no corregir que meter un CFO falso.
        return s

    # Quitar el giro que se acumula de símbolo a símbolo.
    n = np.arange(s.size)
    sin_cfo = s * np.exp(-2j * np.pi * cfo * n)

    # Estimar y quitar la fase constante después de corregir el CFO.
    referencia = np.pi if M == 4 else 0.0
    theta = (np.angle(np.mean(sin_cfo ** M)) - referencia) / M
    return sin_cfo * np.exp(-1j * theta)


RANGO_IMAGEN = 1.5   # la imagen cubre [-1.5, 1.5] en I y en Q
N_BINS = 32          # 32 x 32 píxeles; cada píxel mide 3/32 ≈ 0.094 de lado


def imagen_constelacion(obs, bins=N_BINS, rango=RANGO_IMAGEN):
    """Observación (2, 1024) -> imagen (32, 32) float32: fracción de símbolos por píxel.

    Filas = Q (de abajo hacia arriba con origin="lower"), columnas = I.
    La suma es la fracción de los 118 símbolos que cayó dentro del rango (<= 1).
    """
    s = simbolos_corregidos(obs)

    # histogram2d cuenta cuántos puntos (x, y) caen en cada celda de una rejilla.
    # Entrada: x = [0.1, 0.1, -1.0], y = [0.2, 0.2, 1.0], bins=2, range [[-1.5,1.5]]*2
    # Salida:  [[0, 1],     fila = bin de x, columna = bin de y
    #           [2, 0]]     (los dos primeros puntos caen en x>0, y>0)
    # También devuelve los bordes de los bins en x y en y; aquí no se usan.
    conteo, _, _ = np.histogram2d(
        s.real, s.imag,
        bins=bins,
        range=[[-rango, rango], [-rango, rango]],
    )

    # Dividir entre 118 hace que la imagen no dependa de cuántos símbolos hubo.
    # .T transpone: [bin de I, bin de Q] -> [bin de Q, bin de I], como espera imshow.
    return (conteo / s.size).T.astype(np.float32)