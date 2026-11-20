# Probabilidad de detección: teoría, programación e interpretación

Esta nota explica la evaluación del detector de energía del proyecto de clasificación de modulaciones: qué mide la probabilidad de detección, cómo se calcula con `val` y cómo interpretar las dos curvas de las capturas. Se separan los conceptos de comunicaciones de las operaciones de programación para poder reutilizar el procedimiento en otros proyectos.

Los resultados numéricos se transcriben de las capturas proporcionadas; no corresponden a una nueva ejecución. Las señales evaluadas son **sintéticas**, generadas con ruido y efectos de canal. Aquí «hay señal» significa que existe una transmisión en la observación, no que sea una captura de radio realizada con hardware.

## 1. Qué estamos evaluando

Ya tenemos una regla que recibe una observación de 1024 muestras complejas y responde si detecta señal. Ahora queremos medir cuántas transmisiones consigue detectar cuando cambia su relación con el ruido.

Esta etapa evalúa **presencia de señal**. Todavía no mide si el sistema identifica correctamente BPSK, QPSK u otra modulación.

| Situación real | Respuesta del detector | Nombre del resultado |
|---|---|---|
| Solo ruido | No hay señal | Rechazo correcto |
| Solo ruido | Hay señal | Falsa alarma |
| Señal y ruido | Hay señal | Detección |
| Señal y ruido | No hay señal | Omisión o señal no detectada |

Las dos probabilidades principales son:

$$
P_{\mathrm{fa}}=P(\text{declarar señal}\mid\text{solo ruido})
$$

$$
P_d=P(\text{declarar señal}\mid\text{señal presente}).
$$

La barra vertical significa «dado que». Son probabilidades condicionadas a situaciones diferentes. Por eso las observaciones de ruido sirven para medir falsas alarmas y se excluyen al calcular detecciones.

**Un objetivo de falsas alarmas del 1 % no implica detectar el 99 % de las transmisiones.** Son medidas diferentes.

## 2. Comunicaciones: qué hace el detector

Representamos una observación como:

$$
y[n]=I[n]+jQ[n],\qquad n=0,\ldots,1023.
$$

Calculamos su transformada discreta de Fourier con la FFT:

$$
Y[k]=\sum_{n=0}^{N-1}y[n]e^{-j2\pi kn/N},\qquad N=1024.
$$

La cantidad $|Y[k]|^2$ permite comparar el contenido de potencia entre frecuencias. En la implementación usamos $P[k]=|Y[k]|^2/N$; ese factor común se cancela al formar el cociente.

Para un pulso RRC con $\beta=0.35$ y ocho muestras por símbolo, el borde ideal es:

$$
f_{\mathrm{borde}}=\frac{1+\beta}{2\,\mathrm{sps}}
=\frac{1.35}{16}=0.084375\quad\text{ciclos/muestra}.
$$

Definimos dos conjuntos de índices de frecuencia:

$$
\mathcal K_{\mathrm{dentro}}=\{k:|f_k|\leq0.084375\},
\qquad
\mathcal K_{\mathrm{fuera}}=\{k:|f_k|>0.1\}.
$$

La separación entre ambos deja una zona de guarda. Con esta FFT y estas máscaras hay 173 frecuencias dentro y 819 fuera. Las restantes no participan en el estadístico.

El estadístico es:

$$
T(y)=
\frac{\displaystyle\frac{1}{173}\sum_{k\in\mathcal K_{\mathrm{dentro}}}P[k]}
{\displaystyle\frac{1}{819}\sum_{k\in\mathcal K_{\mathrm{fuera}}}P[k]}.
$$

Usamos **promedios**, porque las zonas tienen distinta cantidad de frecuencias. Bajo ruido blanco, ambas tienen la misma potencia media esperada por frecuencia, de modo que el cociente se concentra cerca de uno. Cuando aparece una señal que concentra energía dentro de la banda, el numerador suele aumentar y el cociente tiende a crecer.

La regla de decisión es:

$$
\widehat H(y)=
\begin{cases}
1,&T(y)>\eta,\\
0,&T(y)\leq\eta,
\end{cases}
\qquad \eta\approx1.208423.
$$

En el notebook este umbral se llama `gamma`. Aquí lo escribimos como $\eta$ para no confundirlo con una variable que represente Es/N0.

### Por qué dejamos el umbral fijo

Bajo ruido complejo gaussiano blanco, usando la FFT completa sin ventana y las zonas disjuntas anteriores, el cociente sigue una distribución F:

$$
T\mid H_0\sim F(346,1638).
$$

El umbral se escogió como el cuantil 0.99 de esa distribución para un objetivo teórico $P_{\mathrm{fa}}=0.01$. Las capturas de esta nota muestran la evaluación de $P_d$, no un nuevo resultado numérico de falsas alarmas.

Mantener el umbral fijo permite medir cómo responde **la misma regla** a condiciones distintas. Si lo cambiamos con cada Es/N0, estaríamos evaluando reglas diferentes y también modificaríamos su comportamiento ante ruido.

En el código mostrado, imprimir `gamma` no basta para cambiar el detector: debemos asegurar que `detectar_senal()` utiliza ese mismo umbral internamente.

### Por qué multiplicar por diez no cambia T

Si $y'[n]=10y[n]$, entonces $Y'[k]=10Y[k]$ y $P'[k]=100P[k]$. Por tanto:

$$
T(y')=\frac{100\,\overline P_{\mathrm{dentro}}}
{100\,\overline P_{\mathrm{fuera}}}=T(y).
$$

Esto explica la invariancia ante una ganancia común aplicada a señal y ruido. No equivale a aumentar solo la señal: en ese caso sí cambia la relación señal/ruido.

## 3. Qué datos utilizamos

Usamos el conjunto de validación `val`:

| Propiedad | Valor |
|---|---|
| Modulaciones | BPSK, QPSK, 8PSK, 16QAM, 64QAM y 16APSK |
| Es/N0 nominal | De −10 a 20 dB, con pasos de 2 dB |
| Niveles nominales | 16 |
| Observaciones por modulación y nivel | 100 |
| Observaciones con señal por nivel | $6\times100=600$ |
| Total de observaciones con señal | $16\times600=9600$ |
| Muestras complejas por observación | 1024 |

Las 1600 observaciones de la clase `ruido` de `val` no entran en este cálculo. Una observación completa produce **una decisión**; las 1024 muestras que contiene no cuentan como 1024 pruebas independientes de detección.

Usar `val` permite estudiar el comportamiento durante el desarrollo. Reservamos `test` para una evaluación final después de fijar las decisiones de diseño.

## 4. Primera curva: P_d frente a Es/N0 nominal

El Es/N0 nominal es el valor solicitado al generador. Para cada nivel contamos cuántas observaciones con señal superan el umbral:

$$
\widehat P_d(e)=\frac{D_e}{N_e}
=\frac{\sum_{i\in\mathcal I_e}\mathbf1\{T(y_i)>\eta\}}
{|\mathcal I_e|}.
$$

Aquí $\mathcal I_e$ contiene las observaciones cuyo nivel nominal es $e$, $D_e$ cuenta las detectadas y $N_e=600$. La función indicadora $\mathbf1$ vale uno si se cumple la condición y cero en caso contrario. El sombrero indica que calculamos una **estimación a partir de datos**.

Por ejemplo, a 0 dB se detectaron 459 de 600:

$$
\widehat P_d(0\ \mathrm{dB})=\frac{459}{600}=0.765=76.5\%.
$$

Las otras 141 observaciones contenían señal, pero el detector no la declaró presente.

### Programación: cargar y validar

Los siguientes bloques usan variables y funciones ya definidas en el notebook: `DATA_DIR`, `N_SAMPLES`, `gamma`, `detectar_senal` y `save_figure`. No constituyen por sí solos un programa independiente. El directorio utilizado por `save_figure` debe apuntar a `figs/deteccion/`.

```python
import csv
import numpy as np
import matplotlib.pyplot as plt

X_val = np.load(
    str(DATA_DIR / "val" / "X.npy"),
    mmap_mode="r",
    allow_pickle=False,
)

with (DATA_DIR / "val" / "meta.csv").open(
    "r", newline="", encoding="utf-8"
) as file:
    metadata_val = list(csv.DictReader(file))

if X_val.ndim != 3 or X_val.shape[1:] != (2, N_SAMPLES):
    raise ValueError("Forma incorrecta de X_val.")

if len(X_val) != len(metadata_val):
    raise ValueError("X_val y meta.csv tienen distintas cantidades de observaciones.")
```

| Operación | Concepto de programación y significado |
|---|---|
| `DATA_DIR / "val" / "X.npy"` | Construcción de una ruta con un objeto `Path`. Aquí `/` une partes de una ruta. |
| `mmap_mode="r"` | Acceso mediante un mapa de memoria de solo lectura; evita cargar necesariamente todo el arreglo de una vez. |
| `allow_pickle=False` | Carga el arreglo numérico sin permitir objetos serializados con pickle. |
| `with ... as file` | Gestor de contexto: cierra el archivo al salir del bloque. |
| `csv.DictReader(file)` | Convierte cada fila del CSV en un diccionario; sus campos se leen inicialmente como texto. |
| `list(...)` | Materializa las filas para poder recorrerlas varias veces. |
| `.ndim` y `.shape` | Comprueban dimensiones y tamaños del arreglo. |
| `.shape[1:]` | Selecciona todos los tamaños salvo el primero: esperamos `(2, 1024)`. |
| `raise ValueError(...)` | Detiene el cálculo con una explicación si la estructura no es válida. |

La fila `metadata_val[i]` debe describir la señal `X_val[i]`. La igualdad de longitudes es necesaria, pero no demuestra por sí sola que las filas estén en el orden correcto: ese orden debe conservarse al generar y guardar los datos.

### Programación: contar observaciones y detecciones

```python
MODULATIONS = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "16APSK")
ESN0_LEVELS = np.arange(-10, 22, 2)

totals = np.zeros(len(ESN0_LEVELS), dtype=np.int64)
detections = np.zeros(len(ESN0_LEVELS), dtype=np.int64)
counts_per_pair = {}

level_index = {
    float(level): i for i, level in enumerate(ESN0_LEVELS)
}

for index, row in enumerate(metadata_val):
    mod = row["mod"]

    if mod == "ruido":
        continue

    if mod not in MODULATIONS:
        raise ValueError("Modulación desconocida: {}".format(mod))

    esn0 = float(row["snr_db"])
    if esn0 not in level_index:
        raise ValueError("Es/N0 inesperado: {}".format(esn0))

    position = level_index[esn0]
    iq = X_val[index]
    observation = iq[0] + 1j * iq[1]

    if not np.all(np.isfinite(observation)):
        raise ValueError("Observación no finita en el índice {}.".format(index))

    totals[position] += 1
    detections[position] += int(detectar_senal(observation))

    pair = (mod, esn0)
    counts_per_pair[pair] = counts_per_pair.get(pair, 0) + 1

for mod in MODULATIONS:
    for level in ESN0_LEVELS:
        if counts_per_pair.get((mod, float(level)), 0) != 100:
            raise ValueError(
                "Se esperaban 100 observaciones para {} a {} dB.".format(mod, level)
            )

pd_observed = detections / totals.astype(np.float64)

print("Umbral fijo gamma: {:.6f}".format(gamma))
for level, detected, total, pd_value in zip(
    ESN0_LEVELS, detections, totals, pd_observed
):
    print("{:>3} dB: {}/{} -> {:.4f}".format(level, detected, total, pd_value))
```

Las operaciones importantes son:

- **`np.arange(-10, 22, 2)`** genera −10, −8, …, 20. El límite superior, 22, queda excluido.
- **`np.zeros`** reserva contadores inicialmente en cero. Usamos enteros porque contamos observaciones.
- **Comprensión de diccionarios:** `level_index` asocia un nivel físico con una posición del arreglo: `-10.0 → 0`, `-8.0 → 1`, etc.
- **`enumerate`** entrega simultáneamente el índice y la fila. Así podemos buscar los metadatos y su señal correspondiente.
- **`continue`** salta al siguiente elemento del bucle cuando la observación es ruido.
- **`float(...)`** transforma el texto del CSV en un número. Aquí comparamos niveles enteros representables exactamente; con niveles decimales arbitrarios conviene revisar la estrategia de comparación.
- **`iq[0] + 1j * iq[1]`** reconstruye el vector complejo a partir de las dos filas I/Q.
- **`np.isfinite` y `np.all`** comprueban que ninguna muestra contiene `NaN` o infinito.
- **`int(True)=1` e `int(False)=0`** convierten la decisión en un valor que se puede sumar. Esta línea implementa la función indicadora de la ecuación.
- **`+= 1`** incrementa el contador existente.
- **Una tupla como clave:** `(mod, esn0)` identifica una combinación de modulación y nivel.
- **`.get(pair, 0)`** recupera el contador o devuelve cero si esa combinación todavía no aparece.
- **Bucles anidados:** revisan todas las combinaciones y verifican que cada una tenga 100 observaciones. Después de esta comprobación, cada denominador es 600.
- **División vectorizada:** calcula los 16 cocientes a la vez.
- **`zip`** recorre arreglos en paralelo para imprimir las columnas relacionadas.
- **`{:.4f}`** muestra cuatro decimales; modifica la presentación, no la precisión del cálculo guardado.

Como cada modulación aporta exactamente 100 observaciones, la curva agregada equivale al promedio de las seis probabilidades estimadas por modulación. Puede ocultar diferencias entre ellas; evaluar cada modulación por separado permitiría detectarlas.

### Resultados registrados

| Es/N0 nominal (dB) | Detectadas | Total | P_d estimada |
|---:|---:|---:|---:|
| −10 | 67 | 600 | 0.1117 |
| −8 | 117 | 600 | 0.1950 |
| −6 | 230 | 600 | 0.3833 |
| −4 | 306 | 600 | 0.5100 |
| −2 | 367 | 600 | 0.6117 |
| 0 | 459 | 600 | 0.7650 |
| 2 | 499 | 600 | 0.8317 |
| 4 | 531 | 600 | 0.8850 |
| 6 | 565 | 600 | 0.9417 |
| 8 | 572 | 600 | 0.9533 |
| 10 | 587 | 600 | 0.9783 |
| 12 | 593 | 600 | 0.9883 |
| 14 | 596 | 600 | 0.9933 |
| 16 | 600 | 600 | 1.0000 |
| 18 | 599 | 600 | 0.9983 |
| 20 | 600 | 600 | 1.0000 |

### Programación: dibujar la primera curva

```python
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    ESN0_LEVELS,
    pd_observed,
    marker="o",
    linewidth=2,
    label="Validación: seis modulaciones, 600 observaciones por punto",
)
ax.set_xlabel("Es/N0 nominal (dB)")
ax.set_ylabel("Probabilidad de detección estimada")
ax.set_title("Detección de señal — umbral fijo gamma = {:.4f}".format(gamma))
ax.set_xticks(ESN0_LEVELS)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)

plt.tight_layout()
save_figure(fig, "07_pd_vs_esn0_val")
plt.show()
```

`fig` es la figura completa y `ax` el área donde dibujamos. `plot(x, y)` relaciona niveles y probabilidades; los marcadores representan resultados calculados y las líneas los unen para facilitar la lectura. `set_xticks` fija las marcas del eje y `set_ylim` deja un pequeño margen sobre uno, sin que una probabilidad pueda superar ese valor. `tight_layout` ajusta los espacios y `save_figure` guarda la figura antes de mostrarla.

Con la función de guardado del proyecto se producen los archivos `07_pd_vs_esn0_val.png` y `07_pd_vs_esn0_val.pdf` en `figs/deteccion/`.

### Cómo leer la curva nominal

A −10 dB se detecta aproximadamente el 11 %; a 0 dB, el 76.5 %; a 10 dB, el 97.83 %. Desde 12 dB los puntos mostrados superan el 98 %.

El canal Rayleigh hace variar la intensidad recibida entre observaciones con el mismo nivel nominal. Esto es consistente con una transición gradual: un nivel nominal alto puede acompañarse de un desvanecimiento que dificulte detectar esa realización. La curva por sí sola no demuestra que el fading sea la única causa de su forma.

La pequeña diferencia entre 16 y 18 dB equivale a una observación. No basta para concluir que el detector empeora al aumentar Es/N0.

## 5. Segunda curva: P_d frente a snr_inst_db

La primera curva agrupa por el nivel solicitado al generador. La segunda usa el valor `snr_inst_db` registrado para cada observación, que pretende describir su condición realizada después de los efectos del canal.

**La definición exacta de este campo debe comprobarse en el generador.** Su nombre no demuestra si representa SNR por muestra, Es/N0 realizado u otra convención. Las capturas lo llaman «Es/N0 instantáneo»; esta nota conserva el nombre del campo y evita asumir una equivalencia no verificada.

Por ejemplo, bajo una normalización de pulso de energía unitaria y las convenciones usuales del modelo discreto:

$$
\mathrm{SNR}_{\mathrm{muestra,dB}}
=(E_s/N_0)_{\mathrm{dB}}-10\log_{10}(\mathrm{sps}).
$$

Para ocho muestras por símbolo, la diferencia sería aproximadamente 9.03 dB. Esta relación ilustra por qué importa revisar la normalización; **no es una corrección que debamos aplicar automáticamente a `snr_inst_db`**.

Las dos curvas usan las mismas señales y el mismo detector. Cambiamos la forma de agrupar sus resultados. Los metadatos se usan para evaluar; no se pasan como información adicional a `detectar_senal()`.

### Por qué agrupamos en intervalos

Los valores realizados son continuos: podrían ser −6.82, −6.14 o −5.37 dB. Si contáramos cada valor exacto por separado, muchos grupos tendrían una sola observación.

Usamos intervalos de ancho 2 dB, por ejemplo $[-8,-6)$ y $[-6,-4)$. El extremo izquierdo se incluye y el derecho se excluye. Estos intervalos del eje de análisis **no son los bins de frecuencia de la FFT**.

Para cada intervalo $B_j$:

$$
\widehat P_{d,j}
=\frac{\sum_i\mathbf1\{s_i\in B_j\}\,\mathbf1\{T(y_i)>\eta\}}
{\sum_i\mathbf1\{s_i\in B_j\}},
$$

donde $s_i$ es el valor registrado en `snr_inst_db`. El denominador cambia entre intervalos y las modulaciones ya no tienen necesariamente el mismo peso dentro de cada uno.

### Programación: reunir valores y decisiones

```python
instant_values = []
instant_decisions = []

for index, row in enumerate(metadata_val):
    if row["mod"] == "ruido":
        continue

    instant_db = float(row["snr_inst_db"])
    if not np.isfinite(instant_db):
        raise ValueError("snr_inst_db no válido en la observación {}.".format(index))

    iq = X_val[index]
    observation = iq[0] + 1j * iq[1]

    instant_values.append(instant_db)
    instant_decisions.append(int(detectar_senal(observation)))

instant_values = np.asarray(instant_values, dtype=np.float64)
instant_decisions = np.asarray(instant_decisions, dtype=np.int64)

if instant_values.size == 0:
    raise ValueError("No se encontraron observaciones con señal.")
```

Las listas crecen con `.append()`. Cada posición conserva una pareja: condición registrada y decisión. `np.asarray` convierte esas listas en arreglos numéricos para operar con ellas. `.size` indica cuántos elementos contiene el arreglo.

Este bloque vuelve a ejecutar el detector sobre las mismas observaciones. Si el detector es determinista y conserva el umbral, las decisiones coinciden con las utilizadas antes; también podrían guardarse una vez y reutilizarse.

### Programación: construir intervalos que cubran los datos

```python
BIN_WIDTH_DB = 2.0

lower = BIN_WIDTH_DB * np.floor(instant_values.min() / BIN_WIDTH_DB)
upper = BIN_WIDTH_DB * (
    np.floor(instant_values.max() / BIN_WIDTH_DB) + 1
)

bin_count = int(round((upper - lower) / BIN_WIDTH_DB))
bin_edges = lower + BIN_WIDTH_DB * np.arange(bin_count + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
```

`min()` y `max()` buscan los extremos observados. `np.floor()` redondea hacia abajo: para un mínimo de −9.3 dB, `2 * floor(-9.3 / 2)` produce −10 dB. Usar `int` en lugar de `floor` no es equivalente con negativos, porque `int` trunca hacia cero.

El límite superior se lleva al siguiente múltiplo de dos para incluir el máximo observado. Necesitamos un borde más que intervalos: cuatro bordes delimitan tres intervalos. Las rebanadas `[:-1]` y `[1:]` seleccionan los bordes izquierdos y derechos; su promedio da los centros usados al dibujar.

### Programación: histogramas y división segura

```python
totals_inst, _ = np.histogram(instant_values, bins=bin_edges)

detections_inst, _ = np.histogram(
    instant_values,
    bins=bin_edges,
    weights=instant_decisions,
)

pd_inst = np.full(len(totals_inst), np.nan, dtype=np.float64)
nonempty = totals_inst > 0

pd_inst[nonempty] = (
    detections_inst[nonempty]
    / totals_inst[nonempty].astype(np.float64)
)

assert totals_inst.sum() == len(instant_values)

for i in np.flatnonzero(nonempty):
    print("[{:.1f}, {:.1f}) | {} / {} | {:.4f}".format(
        bin_edges[i], bin_edges[i + 1],
        int(detections_inst[i]), int(totals_inst[i]), pd_inst[i]
    ))
```

| Instrucción | Qué hace y por qué |
|---|---|
| `np.histogram(..., bins=bin_edges)` | Cuenta cuántas observaciones caen en cada intervalo. |
| `weights=instant_decisions` | Suma ceros y unos en cada intervalo: obtiene cuántas se detectaron. |
| `totals_inst, _ = ...` | Desempaqueta dos resultados; `_` indica que no utilizamos los bordes devueltos porque ya los tenemos. |
| `np.full(..., np.nan)` | Inicializa las probabilidades como desconocidas. |
| `nonempty = totals_inst > 0` | Crea una máscara booleana que identifica denominadores positivos. |
| `pd_inst[nonempty] = ...` | Calcula únicamente donde hay datos y evita dividir entre cero. |
| `assert ...` | Comprueba que todas las observaciones quedaron contabilizadas. |
| `np.flatnonzero(nonempty)` | Devuelve los índices de los intervalos no vacíos para imprimirlos. |

Un intervalo vacío debe quedar como `NaN`: **no tener datos no equivale a obtener cero detecciones entre observaciones existentes**. Al graficar, los `NaN` dejan huecos en la línea.

NumPy incluye el borde derecho en el último intervalo del histograma. Aquí el límite superior se construye por encima del máximo observado, así que ese detalle no cambia los conteos mostrados.

`assert` es útil como comprobación en el notebook. Para una validación obligatoria de un programa distribuido conviene usar un `if` con `raise`, porque Python puede desactivar las aserciones al ejecutarse con optimizaciones.

### Resultados visibles de la segunda curva

La salida de la captura está truncada: la siguiente tabla selecciona filas legibles y no reconstruye las filas ocultas.

| Intervalo de snr_inst_db (dB) | Detectadas | Total | P_d estimada |
|---|---:|---:|---:|
| [−40, −38) | 0 | 1 | 0.0000 |
| [−24, −22) | 1 | 37 | 0.0270 |
| [−22, −20) | 2 | 54 | 0.0370 |
| [−20, −18) | 1 | 75 | 0.0133 |
| [−18, −16) | 2 | 101 | 0.0198 |
| [−16, −14) | 6 | 200 | 0.0300 |
| [−14, −12) | 12 | 279 | 0.0430 |
| [−12, −10) | 17 | 420 | 0.0405 |
| [−10, −8) | 51 | 471 | 0.1083 |
| [−8, −6) | 138 | 568 | 0.2430 |
| [−6, −4) | 345 | 595 | 0.5798 |
| [−4, −2) | 543 | 592 | 0.9172 |
| [−2, 0) | 568 | 568 | 1.0000 |
| [0, 2) | 570 | 570 | 1.0000 |
| [2, 4) | 572 | 572 | 1.0000 |
| [22, 24) | 118 | 118 | 1.0000 |
| [24, 26) | 43 | 43 | 1.0000 |
| [26, 28) | 11 | 11 | 1.0000 |
| [28, 30) | 1 | 1 | 1.0000 |

Por ejemplo, el punto dibujado a −5 dB resume el intervalo $[-6,-4)$:

$$
\widehat P_d=\frac{345}{595}\approx0.5798.
$$

No significa que las 595 observaciones tuvieran exactamente −5 dB; esa coordenada es el centro del intervalo.

### Programación: dibujar probabilidad y cantidad de datos

```python
fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

axes[0].plot(
    bin_centers, pd_inst,
    marker="o", linewidth=2,
    label="Validación: intervalos de 2 dB",
)
axes[0].set_ylabel("Probabilidad de detección estimada")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Detección frente a snr_inst_db — gamma = {:.4f}".format(gamma))
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].bar(
    bin_centers, totals_inst,
    width=0.9 * BIN_WIDTH_DB,
    color="steelblue",
)
axes[1].set_xlabel("snr_inst_db registrado (dB)")
axes[1].set_ylabel("Observaciones")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
save_figure(fig, "08_pd_vs_snr_inst_db_val")
plt.show()
```

La etiqueta de este ejemplo usa el nombre del campo hasta confirmar su definición en el generador. El cálculo es el mismo de las capturas.

`plt.subplots(2, 1)` crea dos gráficos apilados. `sharex=True` alinea el eje horizontal y `height_ratios=[3, 1]` da tres veces más altura al superior. `axes[0]` muestra la probabilidad y `axes[1]` la cantidad de observaciones que sostiene cada punto. El ancho de las barras es el 90 % del intervalo, dejando una pequeña separación visual.

Se guardan `08_pd_vs_snr_inst_db_val.png` y `08_pd_vs_snr_inst_db_val.pdf` en `figs/deteccion/`.

## 6. Cómo comparar ambas curvas

| Pregunta | Curva nominal | Curva con snr_inst_db |
|---|---|---|
| ¿Cómo agrupa? | Por nivel solicitado al generador | Por intervalos del valor realizado registrado |
| ¿Cuántos datos tiene cada punto? | 600 | Una cantidad variable |
| ¿Las modulaciones pesan igual? | Sí, 100 observaciones de cada una | No necesariamente |
| ¿Qué ayuda a estudiar? | Rendimiento bajo cada condición de generación | Relación entre detección y condición realizada |
| ¿Cambia el umbral? | No | No |
| ¿Cambia el detector? | No | No |

La segunda curva tiene una transición más pronunciada: pasa de aproximadamente 24 % en $[-8,-6)$ a 58 % en $[-6,-4)$ y a 92 % en $[-4,-2)$. Es consistente con que agrupar por la condición realizada reúna observaciones más similares en dificultad que agrupar únicamente por el nivel nominal.

Los valores registrados pueden salir del intervalo nominal [−10, 20] dB debido a la realización del canal y, según la implementación, a la convención de medida. Hay que consultar la fórmula del generador para atribuir con precisión cada diferencia.

El histograma inferior es parte esencial de la lectura: una probabilidad de uno basada en una observación tiene mucha menos evidencia que una probabilidad de uno basada en cientos.

## 7. Incertidumbre: qué significa observar 100 %

El valor $\widehat P_d=1$ significa que **todas las observaciones evaluadas en ese grupo fueron detectadas**. No demuestra que toda transmisión futura vaya a detectarse.

Como orientación, para ensayos independientes con una probabilidad común $p$, el error estándar de una proporción es:

$$
\mathrm{SE}(\widehat P_d)\approx\sqrt{\frac{p(1-p)}{N}}.
$$

Cerca de $p=0.5$ y con $N=600$:

$$
\mathrm{SE}\approx\sqrt{\frac{0.5(0.5)}{600}}\approx0.0204.
$$

Un margen normal aproximado del 95 % sería $1.96\times0.0204\approx0.04$: alrededor de cuatro puntos porcentuales. Es una orientación bajo esos supuestos; el conjunto mezcla modulaciones y canales, y ese margen no es una garantía universal.

La aproximación normal funciona mal cerca de cero y uno. Para presentar intervalos de confianza conviene usar, por ejemplo, Wilson o un intervalo binomial exacto. Nunca interpretar un error estándar calculado como cero al sustituir $\widehat p=1$ como ausencia de incertidumbre.

Cuando la señal se vuelve despreciable respecto al ruido, el comportamiento del detector tiende al del caso de ruido bajo las mismas hipótesis; por eso $P_d$ puede aproximarse a $P_{\mathrm{fa}}$ en ese límite. Los pocos datos de los extremos no tienen por qué mostrar exactamente el 1 %.

## 8. Qué hemos comprobado y qué queda por verificar

Las capturas muestran que el código evalúa las seis modulaciones de `val`, mantiene el umbral indicado y calcula una curva nominal creciente en términos generales. La segunda agrupación relaciona las decisiones con `snr_inst_db` y muestra cuántas observaciones respaldan cada punto.

Antes de usar estos resultados como conclusiones generales del sistema conviene:

1. Confirmar en el generador la fórmula y normalización de `snr_inst_db`.
2. Verificar que el umbral que imprime el notebook coincide con el utilizado dentro de `detectar_senal`.
3. Mantener la banda del detector coherente con `sps=8`, `beta=0.35` y las condiciones de frecuencia del experimento. Si cambian las máscaras, cambian sus cantidades de bins y el umbral teórico correspondiente.
4. Revisar las curvas por modulación si queremos conocer diferencias que el promedio conjunto puede ocultar.
5. Realizar la evaluación final sobre `test` una vez fijado el diseño.

## 9. Patrón de programación reutilizable

Este análisis aplica un patrón útil fuera de telecomunicaciones:

1. Cargar datos y comprobar su estructura.
2. Seleccionar las observaciones que responden a la pregunta.
3. Aplicar una regla fija a cada observación.
4. Agrupar por una condición relevante.
5. Contar casos y resultados positivos.
6. Dividir únicamente donde exista un denominador válido.
7. Mostrar la métrica junto con la cantidad de datos que la respalda.
8. Interpretar los resultados dentro de las condiciones evaluadas.

En este proyecto, la regla es el detector; el resultado positivo es «hay señal»; y las condiciones de agrupación son Es/N0 nominal o intervalos de `snr_inst_db`. El mismo patrón sirve para medir alarmas de sensores, fallos detectados en motores o éxito de un robot según distancia, iluminación o carga de batería.
