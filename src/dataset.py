import numpy as np
import csv
from pathlib import Path
from src.generator import modulate,channel,noise_only

# CFO normalizado: diferencia de frecuencia dividida entre fs.
# En train, el extremo provoca aproximadamente 37° de giro
# a lo largo de un fragmento de 1024 muestras.
TRAIN_CFO_RANGE = (-1e-4, 1e-4)

# Exp6 ampliará el rango para evaluar generalización.
EXP6_CFO_RANGE = (-3e-4, 3e-4)

IDEAL_CFO_RANGE = (0.0, 0.0)

# Cualquier orientación inicial de la constelación es posible.
PHASE_RANGE_RAD = (0.0, 2.0 * np.pi)

# Sorteamos log10(A): igual probabilidad por década.
# Al aplicar 10**valor, A queda entre 0.1 y 10.
LOG10_GAIN_RANGE = (-1.0, 1.0)

# Retardos expresados en muestras.
# Con 8 muestras/símbolo, el mayor retardo equivale a T/4.
TRAIN_DELAYS_SAMPLES = (0, 1, 2)

# Potencias medias relativas: ecos progresivamente más débiles.
TRAIN_PATH_POWERS_DB = (0.0, -10.0, -20.0)

# Retardos en muestras (con 8 muestras/símbolo, 4 muestras = T/2).
# Dispersión de retardo rms ≈ 1.46 muestras ≈ 0.18 T, contra ≈ 0.04 T en train.
EXP6_DELAYS_SAMPLES = (0, 2, 4)
EXP6_PATH_POWERS_DB = (0.0, -3.0, -6.0)


CHANNEL_PROFILES = {
    "train": {
        "cfo_range": TRAIN_CFO_RANGE,
        "phase_range_rad": PHASE_RANGE_RAD,
        "log10_gain_range": LOG10_GAIN_RANGE,
        "multipath": {
            "delays_samples": TRAIN_DELAYS_SAMPLES,
            "path_powers_db": TRAIN_PATH_POWERS_DB,
            "status": "defined",
        },
    },

    "exp6": {
        "cfo_range": EXP6_CFO_RANGE,

        # Conservamos fase y ganancia para comparar los canales.
        "phase_range_rad": PHASE_RANGE_RAD,
        "log10_gain_range": LOG10_GAIN_RANGE,

        # Multitrayecto fuerte, con ISI moderada, para el Exp. 6 de generalización.
        "multipath": {
            "delays_samples": EXP6_DELAYS_SAMPLES,
            "path_powers_db": EXP6_PATH_POWERS_DB,
            "status": "defined",
        },
    },

    "ideal": {
        "cfo_range": IDEAL_CFO_RANGE,
        "phase_range_rad": PHASE_RANGE_RAD,
        "log10_gain_range": LOG10_GAIN_RANGE,
        "multipath": {
            "delays_samples": None,
            "path_powers_db": None,
            "status": "none",
        },
    },
}

#Seis modulaciones a clasificar
MODS= (
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "16APSK"
)

#orden fijo de las siete clases 
CLASS_NAMES= MODS + ("ruido",)

# Etiquetas numéricas
CLASS_TO_ID = {
    name : index 
    for index,name in enumerate(CLASS_NAMES)
}

# np.arange excluye el límite superior:
# usamos 22 para incluir +20
ESN0_DB_VALUES = np.arange(-10,22,2,dtype=np.float32)


# Cantidad por cada pareja (modulación, Es/N0).
OBS_PER_PAIR = {
    "train": 500,
    "val": 100,
    "test": 100,
    "exp6": 100,
    "ideal": 100,
}


# train/val/test: misma distribución de canal, realizaciones independientes.
# exp6 e ideal: canales distintos para evaluar generalización (notebook 09).
PROFILE_BY_SPLIT = {
    "train": "train",
    "val": "train",
    "test": "train",
    "exp6": "exp6",
    "ideal": "ideal",
}

# Parámetros compartidos por todas las observaciones.
N_SAMPLES = 1024
SPS = 8
RRC_BETA = 0.35
RRC_SPAN = 10
RANDOM_TIMING = True

""" 
prueba
for split, n in OBS_PER_PAIR.items():
    por_clase = len(ESN0_DB_VALUES) * n
    total = len(CLASS_NAMES) * por_clase
    print(f"{split}: {por_clase} por clase, {total} en total")
"""   

SEEDS= {
    "train":1,
    "val":2,
    "test":3,
    "exp6":4,
    "ideal":5,
}

def save_split(split, X, metadata):
    """Guarda las observaciones y sus metadatos en el mismo orden."""

    if X.ndim != 3 or X.shape[1:] != (2, N_SAMPLES):
        raise ValueError("X debe tener forma (N, 2, 1024).")

    if X.dtype != np.float32:
        raise ValueError("X debe ser de tipo float32.")

    if len(metadata) != len(X):
        raise ValueError(
            "Debe existir una fila de metadatos por observación."
        )

    if any(set(row) != set(META_COLUMNS) for row in metadata):
        raise ValueError("Las columnas de los metadatos no coinciden.")

    folder = DATA_DIR / split
    folder.mkdir(parents=True, exist_ok=True)

    np.save(folder / "X.npy", X, allow_pickle=False)

    with (folder / "meta.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata)

    print("Guardado {}: {}".format(split, X.shape))


def generate_split(split):
    #Un generador aleatorio propio para este conjunto
    seed = SEEDS[split]
    rng = np.random.default_rng(seed)
    n = OBS_PER_PAIR[split]
                                        
    profile_name = PROFILE_BY_SPLIT[split]
    profile = CHANNEL_PROFILES[profile_name]
    multipath = profile["multipath"]
    multipath_status = multipath["status"]

    if multipath_status not in ("defined", "none"):
        raise ValueError(
            "El multitrayecto de {} esta pendiente".format(profile_name)
        )

    # Solo hay desvanecimiento multitrayecto si el perfil lo define.
    fading = multipath_status == "defined"

    per_class = len(ESN0_DB_VALUES) * n
    total = len(CLASS_NAMES) * per_class
    
    #Reservamos espacio para todas las observaciones del conjunto 
    X = np.empty((total, 2, N_SAMPLES), dtype=np.float32)
    metadata = []
    index = 0
    
    #Generamos las seis clases de moduclaciones
    for mod in MODS:
        for snr_db in ESN0_DB_VALUES:
            for _ in range(n):
                x = modulate(
                    mod,N_SAMPLES,rng, 
                    sps=SPS,
                    beta=RRC_BETA,
                    span=RRC_SPAN,
                    random_timing=RANDOM_TIMING
                )
            
            #Nuevos valores del canal para cada observacion
                cfo = rng.uniform(*profile["cfo_range"])
                phase = rng.uniform(*profile["phase_range_rad"])
                amp = 10.0 ** rng.uniform(*profile["log10_gain_range"])
                
                y, channel_meta = channel(
                    x,rng,
                    snr_db=float(snr_db),
                    sps=SPS,
                    cfo=cfo,
                    phase=phase,
                    amp=amp,
                    fading=fading,
                    delays=multipath["delays_samples"],
                    powers_db=multipath["path_powers_db"]
                )

                # Fila 0: I. Fila 1:Q.
                # Al asignar a X, los valores se convierten a float32.
                X[index] = np.stack([y.real, y.imag])
                
                metadata.append({
                    "mod":mod,
                    "snr_db":channel_meta["snr_db"],
                    "snr_inst_db":channel_meta["snr_inst_db"],
                    "cfo":cfo,
                    "phase":phase,
                    "amp":amp,
                    "channel":profile_name,
                    "seed":seed,
                })
                
                index += 1
            
        print("{}: {} lista ({}/{})".format(
            split, mod, index, total
        ), flush=True)
        
    #Clase ruido: tantas observaciones como una modulación
    for _ in range(per_class):
        amp = 10 ** rng.uniform(*profile["log10_gain_range"])
        y = noise_only(N_SAMPLES, rng, amp=amp)
        
        X[index] = np.stack([y.real, y.imag])
        
        metadata.append({
            "mod":"ruido",
            "snr_db": np.nan,
            "snr_inst_db":np.nan,
            "cfo": np.nan,
            "phase": np.nan,
            "amp":amp,
            "channel":"noise_only",
            "seed":seed,
        })
        
        index += 1
        
    if index != total:
        raise RuntimeError("Cantidad de observaciones incorrectas")
    
    return X,metadata
        
            
#Donde se guarda
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

META_COLUMNS = (
    "mod",
    "snr_db",
    "snr_inst_db",
    "cfo",
    "phase",
    "amp",
    "channel",
    "seed",
)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        print("Generando {}...".format(split), flush=True)

        X, metadata = generate_split(split)
        save_split(split, X, metadata)

        # Liberamos las referencias antes del siguiente conjunto.
        del X, metadata





