import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

N_SAMPLES = 1024

FEATURE_NAMES = (
    "abs_C20",
    "C21",
    "abs_C40",
    "C42",
    "abs_C60",
    "C63",
)


def normalize_power(signal):
    """Centrar y normalizar una observacion compleja."""
    x = np.asarray(signal, dtype=np.complex128)

    if x.ndim != 1 or x.size < 6:
        raise ValueError("Se necesita un vector de al menos 6 muestras.")

    if not np.all(np.isfinite(x)):
        raise ValueError("La observacion contiene NaN o infinito.")

    centered = x - np.mean(x)
    power = np.mean(np.abs(centered) ** 2)

    if not np.isfinite(power) or power <= 0:
        raise ValueError("La potencia despues de centrar debe ser positiva.")

    return centered / np.sqrt(power)

def calculate_cumulants(signal):
    """Estimar cumulantes mediante momentos de la observacion normalizada."""
    z = normalize_power(signal)
    z_conj = np.conjugate(z)
    magnitude_squared = np.abs(z) ** 2

    # Momentos de orden 2.
    m20 = np.mean(z ** 2)
    m21 = np.mean(magnitude_squared)

    # Momentos de orden 3: necesarios en las formulas completas de orden 6.
    m30 = np.mean(z ** 3)
    m31 = np.mean(z ** 2 * z_conj)

    # Momentos de orden 4.
    m40 = np.mean(z ** 4)
    m41 = np.mean(z ** 3 * z_conj)
    m42 = np.mean(magnitude_squared ** 2)

    # Momentos de orden 6.
    m60 = np.mean(z ** 6)
    m63 = np.mean(magnitude_squared ** 3)

    c20 = m20
    c21 = m21

    c40 = m40 - 3.0 * m20 ** 2

    c42 = (
        m42
        - np.abs(m20) ** 2
        - 2.0 * m21 ** 2
    )

    c60 = (
        m60
        - 15.0 * m40 * m20
        - 10.0 * m30 ** 2
        + 30.0 * m20 ** 3
    )

    c63 = (
        m63
        - 9.0 * m42 * m21
        - 6.0 * np.real(m41 * np.conjugate(m20))
        - 9.0 * np.abs(m31) ** 2
        - np.abs(m30) ** 2
        + 18.0 * np.abs(m20) ** 2 * m21
        + 12.0 * m21 ** 3
    )

    return {
        "C20": c20,
        "C21": float(np.real(c21)),
        "C40": c40,
        "C42": float(np.real(c42)),
        "C60": c60,
        "C63": float(np.real(c63)),
    }


def extract_features(signal):
    """Convertir los cumulantes en seis caracteristicas reales."""
    c = calculate_cumulants(signal)

    features = np.array(
        [
            np.abs(c["C20"]),
            c["C21"],
            np.abs(c["C40"]),
            c["C42"],
            np.abs(c["C60"]),
            c["C63"],
        ],
        dtype=np.float32
    )

    if not np.all(np.isfinite(features)):
        raise ValueError("Se obtuvieron features no finitas.")

    return features

def check_ideal_symbols():
    bpsk = np.tile(
        np.array([-1.0, 1.0], dtype=np.complex128),
        512
    )

    qpsk = np.tile(
        np.array([1.0, 1j, -1.0, -1j], dtype=np.complex128),
        256
    )

    cases = (
        ("BPSK", bpsk, -2.0, 16.0),
        ("QPSK", qpsk, -1.0, 4.0),
    )

    for name, symbols, expected_c42, expected_c63 in cases:
        c = calculate_cumulants(symbols)

        np.testing.assert_allclose(
            c["C42"], expected_c42, rtol=0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            c["C63"], expected_c63, rtol=0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            c["C21"], 1.0, rtol=0.0, atol=1e-10
        )

        # Las features deben mantenerse ante ganancia y fase constantes.
        transformed = 10.0 * np.exp(1j * 0.7) * symbols

        np.testing.assert_allclose(
            extract_features(symbols),
            extract_features(transformed),
            rtol=1e-6,
            atol=1e-6
        )

        print(
            "{}: C42 = {:.6f}, C63 = {:.6f} — correcto".format(
                name, c["C42"], c["C63"]
            )
        )
        
def process_split(split):
    folder = DATA_DIR / split

    X = np.load(
        str(folder / "X.npy"),
        mmap_mode="r",
        allow_pickle=False
    )

    if X.ndim != 3 or X.shape[1:] != (2, N_SAMPLES):
        raise ValueError(
            "{}: X debe tener forma (N, 2, 1024).".format(split)
        )

    if len(X) == 0:
        raise ValueError("{} esta vacio.".format(split))

    with (folder / "meta.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        metadata_count = sum(1 for row in csv.DictReader(file))

    if metadata_count != len(X):
        raise ValueError(
            "{}: X y meta.csv tienen distinta longitud.".format(split)
        )

    features = np.empty(
        (len(X), len(FEATURE_NAMES)),
        dtype=np.float32
    )

    for index in range(len(X)):
        iq = X[index]
        signal = (
            iq[0].astype(np.float64)
            + 1j * iq[1].astype(np.float64)
        )

        try:
            features[index] = extract_features(signal)
        except ValueError as error:
            raise ValueError(
                "{}: error en observacion {}: {}".format(
                    split, index, error
                )
            )

        if (index + 1) % 10000 == 0:
            print(
                "{}: {} / {}".format(split, index + 1, len(X))
            )

    np.save(
        str(folder / "features.npy"),
        features,
        allow_pickle=False
    )

    # Documentar las columnas y el preprocesamiento.
    description = {
        "columns": list(FEATURE_NAMES),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "source": "X.npy",
        "metadata": "meta.csv",
        "row_order": "Identico a X.npy y meta.csv",
        "preprocessing": {
            "remove_mean": True,
            "power_normalization": "Por observacion, despues de centrar",
            "noise_power_subtraction": False
        },
        "estimator": "Cumulantes por sustitucion de momentos empiricos"
    }

    with (folder / "features_info.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(description, file, indent=2, ensure_ascii=False)

    print(
        "Guardado {}: {}".format(
            folder / "features.npy", features.shape
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extraer cumulantes normalizados del dataset."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Ejecutar solamente las pruebas con simbolos ideales."
    )
    args = parser.parse_args()

    check_ideal_symbols()

    if args.check_only:
        return

    for split in ("train", "val", "test"):
        process_split(split)


if __name__ == "__main__":
    main()
    
