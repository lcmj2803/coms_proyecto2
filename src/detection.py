import numpy as np 
from scipy.stats import f as f_distribution
from src.generator import MODS, constellation, rrc_taps


BETA = 0.35
SPS = 8
BAND_EDGE = (1.0 + BETA)/(2.0 * SPS)
OUTSIDE_EDGE = 0.1

def calcular_T(signal):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=1.0)
    power = np.abs(spectrum)**2/N
    
    inside = np.abs(frequencies) <= BAND_EDGE
    outside = np.abs(frequencies) > OUTSIDE_EDGE

    return np.mean(power[inside]) / np.mean(power[outside])


N_SAMPLES = 1024
P_FA = 0.01

_frequencies = np.fft.fftfreq(N_SAMPLES, d=1.0)
K_INSIDE = int(np.count_nonzero(np.abs(_frequencies) <= BAND_EDGE))
K_OUTSIDE = int(np.count_nonzero(np.abs(_frequencies) > OUTSIDE_EDGE))

# Es/N0 ≈ (T - 1) * FACTOR_SNR_ESPECTRAL.
FACTOR_SNR_ESPECTRAL = SPS * K_INSIDE / float(N_SAMPLES)

# Bajo ruido puro, T ~ F(2*K_INSIDE, 2*K_OUTSIDE)
T_NOISE_DISTRIBUTION = f_distribution(2 * K_INSIDE, 2 * K_OUTSIDE)
GAMMA = float(T_NOISE_DISTRIBUTION.ppf(1.0 - P_FA))


def detectar_senal(signal):
    if np.asarray(signal).shape != (N_SAMPLES,):
        raise ValueError("El umbral corresponde a {} muestras complejas.".format(N_SAMPLES))
    statistic = float(calcular_T(signal))
    if not np.isfinite(statistic):
        raise ValueError("El estadistico no es finito; revisa la observacion.")
    return statistic > GAMMA


# k_a = E|a|^4. constellation() ya deja E|a|^2 = 1.
def _curtosis_constelacion(mod):
    symbols = constellation(mod)
    return float(np.mean(np.abs(symbols) ** 4))


CURTOSIS = {
    mod: _curtosis_constelacion(mod)
    for mod in MODS
}

# Fuera de este intervalo el momento de cuarto orden deja de ser util.
SNR_PISO_DB = -20.0
SNR_TECHO_DB = 40.0
RRC_SPAN = 10


def filtro_acoplado_y_diezmado(signal):
    """Filtra con el RRC de transmisión y elige la fase de símbolo.

    Descarta los bordes en los que el filtro no está lleno y prueba las
    SPS fases de diezmado. Se queda con la de mayor energía media.
    """
    observed = np.asarray(signal)
    if observed.shape != (N_SAMPLES,):
        raise ValueError(
            "Se esperaba un arreglo complejo de {} muestras.".format(N_SAMPLES)
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError("La observacion contiene NaN o infinitos.")

    taps = rrc_taps(BETA, SPS, RRC_SPAN)
    filtered = np.convolve(observed, taps, mode="valid")

    best_phase = 0
    best_energy = -1.0
    for phase in range(SPS):
        symbols = filtered[phase::SPS]
        energy = float(np.mean(np.abs(symbols) ** 2))
        if energy > best_energy:
            best_energy = energy
            best_phase = phase

    return filtered[best_phase::SPS], best_phase


def estimar_snr_m2m4(signal, kurtosis=1.0):
    """Estima Es/N0 en dB con los momentos segundo y cuarto.

    Sobre los símbolos z: M2 = E|z|^2, M4 = E|z|^4.
    Con ruido gaussiano complejo y curtosis k_a,
    M2 = S + N y M4 = k_a S^2 + 4 S N + 2 N^2.
    """
    kurtosis = float(kurtosis)
    if not np.isfinite(kurtosis) or kurtosis >= 2.0:
        raise ValueError("La curtosis debe ser finita y menor que 2.")

    symbols, _phase = filtro_acoplado_y_diezmado(signal)
    power = np.abs(np.asarray(symbols, dtype=np.complex128)) ** 2
    m2 = float(np.mean(power))
    m4 = float(np.mean(power ** 2))

    # 2 M2^2 - M4 = (2 - k_a) S^2. Si no sale positivo, el SNR es muy bajo.
    discriminant = 2.0 * m2 * m2 - m4
    if discriminant <= 0.0:
        return SNR_PISO_DB

    signal_power = np.sqrt(discriminant / (2.0 - kurtosis))
    noise_power = m2 - signal_power
    if noise_power <= 0.0:
        return SNR_TECHO_DB

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    if snr_db < SNR_PISO_DB:
        return SNR_PISO_DB
    if snr_db > SNR_TECHO_DB:
        return SNR_TECHO_DB
    return float(snr_db)


def estimar_snr_espectral(signal):
    """Fuera de banda la potencia por bin es A²·N0; dentro hay ruido más señal.
    La señal aporta Es/SPS por muestra y cae en los K_INSIDE bins, luego T - 1 ≈ (Es/N0)·N_SAMPLES/(SPS·K_INSIDE).
    Es/N0 ≈ (T - 1)·SPS·K_INSIDE/N_SAMPLES; la ganancia A se cancela.
    """
    if np.asarray(signal).shape != (N_SAMPLES,):
        raise ValueError(
            "El umbral corresponde a {} muestras complejas.".format(N_SAMPLES)
        )

    statistic = float(calcular_T(signal))
    if not np.isfinite(statistic):
        raise ValueError("El estadistico no es finito; revisa la observacion.")

    if statistic <= 1.0:
        return SNR_PISO_DB

    snr_linear = (statistic - 1.0) * FACTOR_SNR_ESPECTRAL
    snr_db = 10.0 * np.log10(snr_linear)

    if snr_db < SNR_PISO_DB:
        return SNR_PISO_DB
    if snr_db > SNR_TECHO_DB:
        return SNR_TECHO_DB
    return float(snr_db)

