"""Generación de símbolos para las 6 modulaciones del Proyecto 2 COMII.

Compatible con Python 3.6 (corre también en la Jetson Nano).
"""
import numpy as np

MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK"]


def _psk(M, offset=0.0):
    """M puntos sobre el círculo unitario: a_m = exp(j(2*pi*m/M + offset))."""
    m = np.arange(M)                                  # [0, 1, ..., M-1]
    return np.exp(1j * (2 * np.pi * m / M + offset))  # exp de un complejo = cos + j sin


def _qam(M):
    """Cuadrícula L x L con niveles impares {-(L-1), ..., -1, +1, ..., +(L-1)}."""
    L = int(np.sqrt(M))                        # 16QAM -> L=4, 64QAM -> L=8
    levels = np.arange(-(L - 1), L, 2)         # 16QAM -> [-3, -1, 1, 3]
    I, Q = np.meshgrid(levels, levels)         # todas las combinaciones (I, Q)
    return (I + 1j * Q).ravel()                # matriz L x L -> vector de M puntos


def _apsk16(gamma=2.57):
    """16APSK estilo DVB-S2: 4 puntos en el anillo interior + 12 en el exterior.

    gamma = R2/R1 (relación de radios). 2.57 es el valor de DVB-S2 para tasa 2/3.
    """
    inner = 1.0 * np.exp(1j * (2 * np.pi * np.arange(4) / 4 + np.pi / 4))
    outer = gamma * np.exp(1j * (2 * np.pi * np.arange(12) / 12 + np.pi / 12))
    return np.concatenate([inner, outer])      # une los dos arreglos en uno de 16


def constellation(mod):
    """Devuelve los M puntos de la constelación, normalizados a energía media 1."""
    if mod == "BPSK":
        c = _psk(2)                  # {+1, -1}: igual que 2b-1 de tu notebook
    elif mod == "QPSK":
        c = _psk(4, np.pi / 4)       # rotada 45° para quedar en las diagonales
    elif mod == "8PSK":
        c = _psk(8)
    elif mod == "16QAM":
        c = _qam(16)
    elif mod == "64QAM":
        c = _qam(64)
    elif mod == "16APSK":
        c = _apsk16()
    else:
        raise ValueError("Modulación desconocida: {}".format(mod))

    Es = np.mean(np.abs(c) ** 2)     # energía media por símbolo
    c = c / np.sqrt(Es)              # ahora mean(|c|^2) = 1
    return c.astype(np.complex64)


def random_symbols(mod, n_symbols, rng):
    """n_symbols símbolos equiprobables de la constelación `mod`."""
    c = constellation(mod)
    idx = rng.integers(0, len(c), size=n_symbols)  # índices uniformes en [0, M)
    return c[idx]                                  # "fancy indexing": c[idx[k]] para cada k


# ---------------------------------------------------------------------------
# Paso 2: sobremuestreo y filtro de conformación RRC
# ---------------------------------------------------------------------------

def rrc_taps(beta, sps, span):
    """Coeficientes de un filtro raíz de coseno alzado (RRC), con energía unitaria.

    beta : roll-off (0 < beta <= 1)
    sps  : muestras por símbolo
    span : duración del filtro en símbolos (número de taps = span*sps + 1)
    """
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)  # índices de muestra centrados en 0
    t = n / float(sps)                                     # tiempo en periodos de símbolo (T = 1)

    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            # límite de la fórmula cuando t -> 0
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            # límite cuando el denominador 1 - (4 beta t)^2 -> 0
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h ** 2))  # energía unitaria: sum(h^2) = 1


def upsample(symbols, sps):
    """Inserta sps-1 ceros entre símbolos: [a0, 0, ..., 0, a1, 0, ..., 0, ...]."""
    x = np.zeros(len(symbols) * sps, dtype=np.complex64)
    x[::sps] = symbols       # rebanada con paso sps: posiciones 0, sps, 2*sps, ...
    return x


def pulse_shape(symbols, sps, beta=0.35, span=10):
    """Sobremuestrea y filtra con RRC. Devuelve la convolución completa ("full")."""
    h = rrc_taps(beta, sps, span)
    return np.convolve(upsample(symbols, sps), h).astype(np.complex64)


def modulate(mod, n_samples, rng, sps=8, beta=0.35, span=10, random_timing=True):
    """Señal banda base de n_samples muestras, sin transitorios de borde.

    Genera símbolos de sobra y recorta el tramo central, donde el filtro
    ya está "lleno" de símbolos por ambos lados.

    random_timing=True desplaza el inicio entre 0 y sps-1 muestras: el receptor
    no sabe dónde empieza cada símbolo (error de temporización).
    """
    n_sym = n_samples // sps + 2 * span + 1      # símbolos de sobra para los bordes
    s = random_symbols(mod, n_sym, rng)
    x = pulse_shape(s, sps, beta, span)
    start = span * sps                           # salta el transitorio de arranque
    if random_timing:
        start += int(rng.integers(0, sps))
    return x[start:start + n_samples]


# ---------------------------------------------------------------------------
# Paso 3: canal inalámbrico virtual
# Convenio: Es/N0 con N0 = 1. La señal sale de modulate() con Es = 1 y se
# escala por sqrt(10^(snr_db/10)); el ruido siempre tiene varianza 1.
# ---------------------------------------------------------------------------

def awgn(n_samples, rng):
    """Ruido complejo gaussiano circular de varianza 1: E[|w|^2] = 1."""
    w = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    return (w / np.sqrt(2.0)).astype(np.complex64)   # cada componente aporta 1/2


def rayleigh_multipath(x, rng, delays=(0, 1, 2), powers_db=(0.0, -10.0, -20.0)):
    """Canal multitrayecto con taps Rayleigh, constante durante la observación.

    delays    : retardo de cada trayecto, en muestras
    powers_db : potencia media de cada trayecto (perfil de retardo)
    La suma de potencias medias se normaliza a 1: en promedio no cambia Es,
    pero cada realización sí (eso es el fading).
    """
    p = 10.0 ** (np.asarray(powers_db) / 10.0)
    p = p / np.sum(p)
    g = rng.standard_normal(len(p)) + 1j * rng.standard_normal(len(p))
    taps = np.sqrt(p / 2.0) * g                      # E[|tap_l|^2] = p_l
    h = np.zeros(max(delays) + 1, dtype=np.complex64)
    h[list(delays)] = taps
    y = np.convolve(x, h)[:len(x)]                   # mismas muestras que la entrada
    return y.astype(np.complex64), h


def cfo_phase(x, cfo=0.0, phase=0.0):
    """Multiplica por exp(j(2*pi*cfo*n + phase)). cfo en ciclos por muestra (f/fs)."""
    n = np.arange(len(x))
    return (x * np.exp(1j * (2.0 * np.pi * cfo * n + phase))).astype(np.complex64)


def channel(x, rng, snr_db, sps=8, cfo=0.0, phase=0.0, amp=1.0, fading=True,
            delays=(0, 1, 2), powers_db=(0.0, -10.0, -20.0)):
    """Señal limpia (Es = 1) -> fading -> CFO/fase -> AWGN -> ganancia del receptor.

    Devuelve (y, meta). meta["snr_inst_db"] es el Es/N0 real de esta
    realización, después del fading.
    """
    s = x * np.sqrt(10.0 ** (snr_db / 10.0))         # Es = 10^(snr/10), N0 = 1
    if fading:
        s, _ = rayleigh_multipath(s, rng, delays, powers_db)
    s = cfo_phase(s, cfo, phase)
    es_inst = sps * np.mean(np.abs(s) ** 2)          # energía por símbolo medida
    y = amp * (s + awgn(len(s), rng))                # la ganancia no cambia el SNR
    meta = {"snr_db": snr_db,
            "snr_inst_db": float(10.0 * np.log10(es_inst)),
            "cfo": cfo, "phase": phase, "amp": amp, "fading": fading}
    return y.astype(np.complex64), meta


def noise_only(n_samples, rng, amp=1.0):
    """Observación sin transmisión: solo ruido con el mismo piso (N0 = 1)."""
    return (amp * awgn(n_samples, rng)).astype(np.complex64)