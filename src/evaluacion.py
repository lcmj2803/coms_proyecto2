"""Carga un conjunto, predice con el bosque y la CNN, y resume métricas.

Los modelos llegan ya cargados. Este módulo no llama a joblib ni a torch.load.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.features import FEATURE_NAMES, extract_features
from src.pipeline import iq_a_complejo
from src.pre_procesamiento import CLASSES, normalize_iq_batch

N_MUESTRAS = 1024
N_CLASES = len(CLASSES)


def _exigir_iq(iq):
    """Forma (N, 2, 1024). N puede ser 0; el llamador decide si eso basta."""
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[1:] != (2, N_MUESTRAS):
        raise ValueError(
            "Se esperaba iq con forma (N, 2, {}), llegó {}.".format(
                N_MUESTRAS, iq.shape
            )
        )
    return iq


def cargar_conjunto(carpeta):
    """Lee X.npy y meta.csv de la carpeta que llega como argumento.

    Entrada:
        carpeta: ruta de un conjunto (train, val o test).
        Dentro debe haber X.npy y meta.csv. No se busca la raíz
        del proyecto.

    Salida:
        iq:    (N, 2, 1024) float32. Fila 0 = I, fila 1 = Q.
        y_true: (N,) str. El nombre de la clase, columna "mod".
        esn0:  (N,) float64. Es/N0 en dB. NaN donde y_true es "ruido".

    Ejemplo:
        meta.csv
            mod,snr_db
            BPSK,-10
            ruido,nan
        -> y_true = ["BPSK", "ruido"]
           esn0   = [-10.0, nan]
    """
    carpeta = Path(carpeta)

    # np.load lee un arreglo .npy.
    # allow_pickle=False rechaza objetos Python guardados en el archivo.
    # mmap_mode="r" no copia las señales a RAM: las lee del disco al usarlas.
    # El arreglo queda solo lectura y sigue siendo (N, 2, 1024) float32.
    iq = np.load(
        carpeta / "X.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if iq.ndim != 3 or iq.shape[1:] != (2, N_MUESTRAS):
        raise ValueError(
            "X.npy debe tener forma (N, 2, {}); llegó {}.".format(
                N_MUESTRAS, iq.shape
            )
        )
    if iq.dtype != np.float32:
        raise ValueError(
            "X.npy debe ser float32; llegó {}.".format(iq.dtype)
        )

    # csv.DictReader usa la primera línea como nombres de columna.
    # Entrada (texto): "mod,snr_db\nBPSK,-10\nruido,nan\n"
    # Salida: [{"mod": "BPSK", "snr_db": "-10"},
    #          {"mod": "ruido", "snr_db": "nan"}]
    # Los valores salen como str. float("nan") es nan, no un error.
    with (carpeta / "meta.csv").open(
        "r", newline="", encoding="utf-8"
    ) as archivo:
        filas = list(csv.DictReader(archivo))

    if len(filas) != len(iq):
        raise ValueError(
            "meta.csv tiene {} filas y X.npy tiene {}.".format(
                len(filas), len(iq)
            )
        )
    if filas and not {"mod", "snr_db"} <= set(filas[0]):
        raise ValueError(
            "meta.csv debe traer las columnas mod y snr_db."
        )

    y_true = np.array([fila["mod"] for fila in filas])
    esn0 = np.array(
        [float(fila["snr_db"]) for fila in filas],
        dtype=np.float64,
    )
    # El ruido no tiene Es/N0. Se fuerza NaN aunque el CSV traiga un número.
    esn0[y_true == "ruido"] = np.nan

    return iq, y_true, esn0


def predecir_rf(rf, iq):
    """Cumulantes de cada observación, sin C21, y probabilidades en orden CLASSES.

    Entrada:
        rf: RandomForestClassifier ya entrenado. No se llama a fit.
        iq: (N, 2, 1024). Cada fila es una observación I/Q.

    Salida:
        nombres: (N,) str. Lo que devuelve rf.predict.
        prob:    (N, 7) float. La columna i es P(CLASSES[i] | fila).
                 CLASSES = (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 16APSK, ruido).
                 No es el orden alfabético de rf.classes_.

    Ejemplo, con tres clases para ver el reorden:
        rf.classes_ = ["8PSK", "BPSK", "QPSK"]
        una fila de predict_proba = [0.10, 0.70, 0.20]
        CLASSES de este ejemplo   = ("BPSK", "QPSK", "8PSK")
        -> prob de esa fila       = [0.70, 0.20, 0.10]
        La columna 0 pasa a ser BPSK.
    """
    iq = _exigir_iq(iq)

    # iq_a_complejo((2, 1024)) junta I y Q en un vector complejo de 1024.
    # extract_features(ese vector) devuelve 6 float32, el orden de FEATURE_NAMES:
    # abs_C20, C21, abs_C40, C42, abs_C60, C63.
    features = np.empty((len(iq), len(FEATURE_NAMES)), dtype=np.float32)
    for indice, obs in enumerate(iq):
        features[indice] = extract_features(iq_a_complejo(obs))

    if features.shape != (len(iq), len(FEATURE_NAMES)):
        raise ValueError(
            "Se esperaban features (N, {}); llegó {}.".format(
                len(FEATURE_NAMES), features.shape
            )
        )

    # np.delete(arreglo, indice, axis=1) quita una columna y devuelve copia.
    # Entrada: np.delete([[1.0, 9.0, 3.0]], 1, axis=1)
    # Salida:  [[1.0, 3.0]]
    # El índice de C21 se busca por nombre. No se escribe el 1 a mano.
    c21 = FEATURE_NAMES.index("C21")
    X = np.delete(features, c21, axis=1)
    if X.shape != (len(iq), len(FEATURE_NAMES) - 1):
        raise ValueError(
            "El bosque espera (N, {}); llegó {}.".format(
                len(FEATURE_NAMES) - 1, X.shape
            )
        )

    # predict(X) con X de forma (N, 5) devuelve N nombres de clase.
    # No reordena nada: el nombre ya es la etiqueta, no un índice de columna.
    nombres = np.asarray(rf.predict(X))
    if nombres.shape != (len(iq),):
        raise ValueError(
            "rf.predict debía devolver (N,); llegó {}.".format(nombres.shape)
        )

    # predict_proba(X) devuelve P(clase | fila). No se llama a fit.
    # La forma es (N, n_clases). La columna j corresponde a rf.classes_[j],
    # que está en orden alfabético, no en el de CLASSES.
    proba = np.asarray(rf.predict_proba(X))
    if proba.ndim != 2 or proba.shape[0] != len(iq):
        raise ValueError(
            "rf.predict_proba debía devolver (N, n_clases); llegó {}.".format(
                proba.shape
            )
        )
    if proba.shape[1] != len(rf.classes_):
        raise ValueError(
            "predict_proba tiene {} columnas y rf.classes_ tiene {}.".format(
                proba.shape[1], len(rf.classes_)
            )
        )

    # list.index(nombre) busca la posición de ese nombre en rf.classes_.
    # Con el ejemplo del docstring: "BPSK" -> 1, "QPSK" -> 2, "8PSK" -> 0.
    # proba[:, orden] toma esas columnas y las deja en el orden de CLASSES.
    orden = [list(rf.classes_).index(nombre) for nombre in CLASSES]
    prob = proba[:, orden]
    if prob.shape != (len(iq), N_CLASES):
        raise ValueError(
            "prob debía quedar (N, {}); llegó {}.".format(N_CLASES, prob.shape)
        )

    return nombres, prob


def predecir_cnn(cnn, iq, batch=256):
    """Probabilidades de la CNN por lotes, en el orden de CLASSES.

    Entrada:
        cnn: ModulationCNN ya cargada. No se llama a torch.load.
        iq: (N, 2, 1024) float.
        batch: entero >= 1. Cuántas observaciones entran juntas a la red.
               El último lote puede ser más corto.

    Salida:
        nombres: (N,) str. CLASSES[argmax de la fila].
        prob:    (N, 7) float32. La columna i es P(CLASSES[i] | fila).
                 Cada fila suma 1.

    Ejemplo:
        puntuaciones de una fila = [2.0, 0.0, ...]
        softmax de esa fila      ≈ [0.88, 0.12, ...]   (suman 1)
        argmax = 0 -> nombres = "BPSK", porque CLASSES[0] es BPSK.
    """
    iq = _exigir_iq(iq)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(
            "batch debe ser un entero >= 1; llegó {}.".format(batch)
        )

    # eval() deja la red en inferencia: sin dropout y con la estadística
    # fija de BatchNorm. No entrena.
    cnn.eval()
    trozos = []

    # torch.no_grad() no guarda gradientes. La red no se entrena y
    # la pasada gasta menos memoria.
    with torch.no_grad():
        for inicio in range(0, len(iq), batch):
            # normalize_iq_batch centra I y Q y deja potencia 1.
            # Devuelve un float32 nuevo; no modifica iq.
            lote = normalize_iq_batch(iq[inicio:inicio + batch])
            if lote.ndim != 3 or lote.shape[1:] != (2, N_MUESTRAS):
                raise ValueError(
                    "El lote normalizado debía ser (B, 2, {}); llegó {}.".format(
                        N_MUESTRAS, lote.shape
                    )
                )

            # torch.from_numpy comparte memoria con el arreglo; no copia.
            # El lote ya es contiguo y escribible, así que from_numpy lo acepta.
            puntuaciones = cnn(torch.from_numpy(lote))

            # softmax(puntuaciones, dim=1) las vuelve probabilidades por fila.
            # dim=1 es el eje de las clases, no el de las observaciones.
            # Entrada: [[2.0, 0.0]] -> Salida: [[0.88, 0.12]] (suman 1).
            # La columna i es la probabilidad de CLASSES[i].
            probabilidades = torch.softmax(puntuaciones, dim=1)
            if probabilidades.ndim != 2 or probabilidades.shape[1] != N_CLASES:
                raise ValueError(
                    "La CNN debía devolver (B, {}); llegó {}.".format(
                        N_CLASES, tuple(probabilidades.shape)
                    )
                )

            # .numpy() pasa el tensor a un arreglo NumPy.
            trozos.append(probabilidades.numpy())

    if not trozos:
        raise ValueError("iq no tiene observaciones.")

    # np.concatenate pega los lotes, uno debajo del otro.
    # astype(float32) deja el dtype pedido. copy=False no copia
    # si el arreglo ya era float32.
    prob = np.concatenate(trozos).astype(np.float32, copy=False)
    if prob.shape != (len(iq), N_CLASES):
        raise ValueError(
            "prob debía quedar (N, {}); llegó {}.".format(N_CLASES, prob.shape)
        )

    # argmax(axis=1) es el índice de la clase más alta de cada fila.
    # Entrada: [[0.1, 0.9, 0.0]] -> Salida: [1].
    # CLASSES es una tupla: CLASSES[np.array([0, 1])] lanza TypeError.
    # np.array(CLASSES)[indices] sí devuelve los nombres.
    indices = np.argmax(prob, axis=1)
    nombres = np.array(CLASSES)[indices]
    if nombres.shape != (len(iq),):
        raise ValueError(
            "nombres debía quedar (N,); llegó {}.".format(nombres.shape)
        )

    return nombres, prob


def metricas(y_true, y_pred):
    """Exactitud, F1 macro, detalle por clase y matriz de confusión 7x7.

    Entrada:
        y_true: (N,) nombres reales. Cada uno tiene que estar en CLASSES.
        y_pred: (N,) nombres predichos, el mismo largo.

    Salida:
        dict con cuatro claves:
            exactitud: float. Aciertos / N.
            f1_macro:  float. Media sin ponderar del F1 de las 7 clases.
            por_clase: dict en el orden de CLASSES. Cada clase tiene
                       precision, recall, f1 (float) y soporte (int).
            matriz:    (7, 7) int. Filas = real, columnas = predicha,
                       las dos en el orden de CLASSES.

    Ejemplo:
        y_true = ["BPSK", "QPSK"]
        y_pred = ["QPSK", "QPSK"]
        labels = ["BPSK", "QPSK"]   (aquí solo se muestran esas dos)
        matriz = [[0, 1],
                  [0, 1]]
        exactitud = 0.5
        La diagonal son aciertos. Con las siete clases, el resto de
        la matriz 7x7 queda en cero y el orden sigue siendo el de CLASSES,
        no el alfabético.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "y_true y y_pred deben ser vectores; llegaron {} y {}.".format(
                y_true.shape, y_pred.shape
            )
        )
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true tiene {} filas y y_pred tiene {}.".format(
                len(y_true), len(y_pred)
            )
        )
    if len(y_true) == 0:
        raise ValueError("No hay observaciones para calcular métricas.")

    desconocidas = sorted(
        (set(y_true.tolist()) | set(y_pred.tolist())) - set(CLASSES)
    )
    if desconocidas:
        raise ValueError(
            "Hay etiquetas fuera de CLASSES: {}.".format(desconocidas)
        )

    # labels fija el orden de filas, columnas y del promedio macro.
    # Sin labels, sklearn ordena las etiquetas que aparezcan, y el
    # bosque es alfabético. list(CLASSES) no es sorted(CLASSES).
    etiquetas = list(CLASSES)

    # precision_recall_fscore_support devuelve cuatro vectores, uno
    # por etiqueta, en el orden de labels:
    #   precision, recall, f1, soporte.
    # zero_division=0 escribe 0.0 si una clase no recibió predicciones,
    # en vez de avisar y dejar el valor indefinido.
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=etiquetas,
        zero_division=0,
    )
    if len(precision) != N_CLASES:
        raise ValueError(
            "Se esperaban {} clases; precision tiene {}.".format(
                N_CLASES, len(precision)
            )
        )

    por_clase = {}
    for nombre, p, r, f, s in zip(etiquetas, precision, recall, f1, soporte):
        por_clase[nombre] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "soporte": int(s),
        }

    # confusion_matrix cuenta pares (real, predicha).
    # Entrada: y_true=["BPSK","QPSK"], y_pred=["QPSK","QPSK"],
    #          labels=["BPSK","QPSK"]
    # Salida:  [[0, 1],
    #           [0, 1]]
    # La diagonal son aciertos.
    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas)
    if matriz.shape != (N_CLASES, N_CLASES):
        raise ValueError(
            "La matriz debía ser ({0}, {0}); llegó {1}.".format(
                N_CLASES, matriz.shape
            )
        )

    # accuracy_score es la fracción de filas en las que y_pred == y_true.
    # f1_score(..., average="macro") es la media del F1 de cada clase,
    # sin ponderar por el soporte. labels incluye las siete aunque
    # alguna no aparezca en este subconjunto.
    return {
        "exactitud": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=etiquetas,
                average="macro",
                zero_division=0,
            )
        ),
        "por_clase": por_clase,
        "matriz": matriz,
    }
