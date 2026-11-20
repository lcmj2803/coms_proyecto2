# Cumulantes y Random Forest: programación y comunicaciones

Guía de estudio de `features.py` y `04_features.ipynb` para el proyecto de clasificación de modulaciones.

Esta nota reúne las dos explicaciones del chat, relacionando las operaciones de Python con la teoría y la interpretación de los resultados. Al final incluye el código de los dos archivos como referencia. Los resultados de clasificación proceden de las salidas guardadas del notebook; no se realizó un nuevo entrenamiento para preparar este documento.

## 1. El recorrido de los datos

| Etapa | Entrada | Salida | Responsabilidad |
|---|---|---|---|
| Generación del dataset | Símbolos, pulso, canal y ruido | Observaciones I/Q | Crear señales y etiquetas conocidas. |
| `features.py` | Una observación de 1024 muestras complejas | Seis características reales | Resumir propiedades estadísticas de la señal. |
| `04_features.ipynb` | Características y etiquetas | Modelo y evaluación en validación | Aprender a distinguir siete categorías. |

Una observación se almacena como dos filas de 1024 números: I y Q. Al reconstruirla:

$$
x[n]=I[n]+jQ[n],\qquad n=0,\ldots,1023.
$$

El extractor convierte esa observación en:

$$
\mathbf f=[|C_{20}|,C_{21},|C_{40}|,C_{42},|C_{60}|,C_{63}].
$$

El notebook elimina `C21`, que queda aproximadamente constante por la normalización, y entrena con las otras cinco características.

**Una fila representa una observación completa.** Las 1024 muestras de esa observación no son 1024 ejemplos independientes de entrenamiento.

---

# Parte I. Entender `features.py`

## 2. Organización del archivo

| Función | Construcción de programación | Papel en el proyecto |
|---|---|---|
| `normalize_power(signal)` | Recibe un vector y devuelve otro. | Centra y normaliza la potencia. |
| `calculate_cumulants(signal)` | Calcula valores intermedios y devuelve un diccionario. | Estima cumulantes de órdenes 2, 4 y 6. |
| `extract_features(signal)` | Selecciona resultados y construye un arreglo. | Obtiene seis números reales. |
| `check_ideal_symbols()` | Construye ejemplos y verifica valores esperados. | Comprueba BPSK, QPSK e invariancias. |
| `process_split(split)` | Carga, recorre y guarda datos. | Procesa un conjunto completo. |
| `main()` | Interpreta opciones de terminal y coordina llamadas. | Ejecuta pruebas y, si corresponde, extracción. |

`def` define una función, pero su cuerpo se ejecuta cuando la llamas. En `extract_features(signal)`, `signal` es el argumento entregado a la función; `return` devuelve su resultado. Las variables locales, como `centered`, pertenecen a esa ejecución de la función.

### Rutas y configuración

```python
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
```

`__file__` identifica el archivo ejecutado. `.resolve()` obtiene una ruta absoluta y cada `.parent` sube un nivel. Si el archivo está en `coms_proyecto2/src/features.py`, la raíz obtenida es `coms_proyecto2`. El operador `/` de `Path` une componentes de rutas.

`FEATURE_NAMES` es una tupla que fija los nombres y el orden de las columnas. Ese orden debe conservarse cuando se entrena el modelo y cuando se utiliza para predecir.

## 3. Centrar y normalizar la potencia

```python
x = np.asarray(signal, dtype=np.complex128)
centered = x - np.mean(x)
power = np.mean(np.abs(centered) ** 2)
return centered / np.sqrt(power)
```

`np.asarray` convierte la entrada en un arreglo. `complex128` representa cada número complejo con dos componentes de doble precisión. Las operaciones actúan sobre todos los elementos del vector.

Matemáticamente:

$$
\bar x=\frac1N\sum_nx[n],\qquad x_c[n]=x[n]-\bar x,
$$

$$
\widehat P=\frac1N\sum_n|x_c[n]|^2,
\qquad z[n]=\frac{x_c[n]}{\sqrt{\widehat P}}.
$$

El vector resultante tiene media aproximadamente cero y potencia media aproximadamente uno:

$$
\frac1N\sum_nz[n]\approx0,
\qquad \frac1N\sum_n|z[n]|^2\approx1.
$$

Dividimos por la raíz de la potencia porque estamos modificando amplitudes y la potencia depende del cuadrado de la amplitud.

**Ejemplo ilustrativo:** para `[-2, 2, -2, 2]`, la media es cero y la potencia es cuatro. Dividir entre dos produce `[-1, 1, -1, 1]`. Este ejemplo explica la operación; la función real exige al menos seis muestras.

El código también verifica:

- `x.ndim == 1`: debe ser un vector.
- `x.size >= 6`: mínimo exigido por esta implementación; no garantiza precisión estadística.
- `np.all(np.isfinite(x))`: no debe contener `NaN` o infinitos.
- `power > 0`: evita dividir entre cero.

Una señal constante queda en cero al restar su media y se rechaza. La normalización ajusta conjuntamente señal y ruido; **no separa ni elimina el ruido**.

## 4. Momentos: calcular promedios de potencias

```python
z_conj = np.conjugate(z)
magnitude_squared = np.abs(z) ** 2
```

Si $z=I+jQ$, su conjugado es $z^*=I-jQ$. Por tanto:

$$
zz^*=I^2+Q^2=|z|^2.
$$

En la convención del archivo, el momento empírico es:

$$
\widehat M_{pq}=\frac1N\sum_nz[n]^{p-q}(z[n]^*)^q.
$$

$p$ representa el orden total y $q$ el número de factores conjugados. El sombrero indica una estimación calculada a partir de muestras.

| Código | Momento | Interpretación |
|---|---|---|
| `np.mean(z ** 2)` | $\widehat M_{20}$ | Promedio del cuadrado complejo; conserva dependencia de fase. |
| `np.mean(magnitude_squared)` | $\widehat M_{21}$ | Potencia media. |
| `np.mean(z ** 4)` | $\widehat M_{40}$ | Momento complejo de cuarto orden. |
| `np.mean(magnitude_squared ** 2)` | $\widehat M_{42}$ | Promedio de $|z|^4$. |
| `np.mean(magnitude_squared ** 3)` | $\widehat M_{63}$ | Promedio de $|z|^6$. |

`z ** 4` eleva cada muestra a la cuarta potencia y `np.mean` promedia el resultado: son operaciones vectorizadas. El programa también calcula `m30`, `m31` y `m41`, necesarios para formar los cumulantes posteriores.

## 5. Cumulantes: combinar los momentos

Los cumulantes combinan momentos descontando contribuciones de órdenes inferiores. Pueden diferenciar distribuciones de símbolos que comparten la misma potencia.

Las expresiones implementadas, sobre muestras previamente centradas, son:

$$
\widehat C_{20}=\widehat M_{20},\qquad
\widehat C_{21}=\widehat M_{21},
$$

$$
\widehat C_{40}=\widehat M_{40}-3\widehat M_{20}^2,
$$

$$
\widehat C_{42}=\widehat M_{42}-|\widehat M_{20}|^2-2\widehat M_{21}^2,
$$

$$
\widehat C_{60}=\widehat M_{60}-15\widehat M_{40}\widehat M_{20}
-10\widehat M_{30}^2+30\widehat M_{20}^3,
$$

$$
\begin{aligned}
\widehat C_{63}={}&\widehat M_{63}-9\widehat M_{42}\widehat M_{21}
-6\operatorname{Re}(\widehat M_{41}\widehat M_{20}^*)\\
&-9|\widehat M_{31}|^2-|\widehat M_{30}|^2
+18|\widehat M_{20}|^2\widehat M_{21}+12\widehat M_{21}^3.
\end{aligned}
$$

Los coeficientes pertenecen a las relaciones matemáticas entre momentos y cumulantes; no son pesos aprendidos por el modelo.

Restar la media no obliga a que los momentos de tercer orden de una muestra finita sean cero. Por eso la implementación conserva esos términos en las expresiones de sexto orden.

### Derivación de C42 para BPSK y QPSK

Para BPSK ideal equilibrada, $z\in\{-1,+1\}$:

$$
M_{20}=1,\quad M_{21}=1,\quad M_{42}=1,
\qquad C_{42}=1-1-2=\boxed{-2}.
$$

Para QPSK ideal equilibrada, $z\in\{1,j,-1,-j\}$, los cuadrados son $1,-1,1,-1$, cuyo promedio es cero:

$$
M_{20}=0,\quad M_{21}=1,\quad M_{42}=1,
\qquad C_{42}=1-0-2=\boxed{-1}.
$$

Ambas constelaciones tienen potencia uno, pero sus cumulantes son distintos. Esa diferencia es una posible fuente de información para el clasificador.

### Qué relación tienen con el ruido

Los cumulantes poblacionales de orden mayor que dos de un proceso gaussiano son cero. Para componentes independientes, los cumulantes se suman. Estas propiedades motivan su uso en clasificación de modulaciones, pero no significan que una estimación finita elimine perfectamente el ruido.

Tu implementación sustituye momentos teóricos por promedios muestrales: es un estimador por sustitución, no una corrección insesgada para muestras finitas. Además, divide por la potencia total observada, que incluye ruido. Por eso el ruido todavía afecta las características normalizadas.

## 6. Construir el vector de características

`calculate_cumulants` devuelve un diccionario. Por ejemplo, `c["C42"]` recupera el valor asociado a esa clave.

`extract_features` construye un arreglo de seis números reales. Aplica `np.abs` a `C20`, `C40` y `C60`, y mantiene el signo de `C42` y `C63`.

Para una transformación constante $y[n]=Ae^{j\phi}x[n]$, con $A>0$:

- La normalización elimina la ganancia común $A$.
- Los cumulantes complejos pueden rotar con la fase; sus magnitudes eliminan esa rotación.
- Los cumulantes con igual cantidad de factores conjugados y no conjugados, como $C_{42}$ y $C_{63}$, permanecen invariantes a la fase constante.

**Esto no equivale a inmunidad al CFO ni al multitrayecto.** El CFO introduce una fase dependiente de la muestra y el multitrayecto suma copias retrasadas, modificando las estadísticas de otra manera.

Los cálculos usan doble precisión y las características se almacenan como `float32` para reducir el tamaño de los archivos.

Como $C_{21}\approx1$ tras normalizar, esa columna sirve de comprobación, pero ofrece poca información para discriminar clases. El notebook la elimina.

## 7. Las pruebas con símbolos ideales

`np.tile` repite una secuencia. BPSK repite dos símbolos 512 veces y QPSK cuatro símbolos 256 veces: ambos casos producen 1024 símbolos equilibrados, sin variación aleatoria en sus proporciones.

| Caso | C21 esperado | C42 esperado | C63 esperado |
|---|---:|---:|---:|
| BPSK ideal | 1 | −2 | 16 |
| QPSK ideal | 1 | −1 | 4 |

`np.testing.assert_allclose` compara resultados con tolerancias. Con `rtol=0` y `atol=1e-10`, exige una diferencia absoluta no superior a $10^{-10}$.

La transformación de prueba es:

```python
transformed = 10.0 * np.exp(1j * 0.7) * symbols
```

Aplica ganancia diez y fase constante de 0.7 radianes. Después se comparan las seis características originales y transformadas.

En la revisión anterior del chat se ejecutó únicamente `--check-only` y las comprobaciones pasaron:

```text
BPSK: C42 = -2.000000, C63 = 16.000000 — correcto
QPSK: C42 = -1.000000, C63 = 4.000000 — correcto
```

Estas pruebas verifican casos conocidos, no garantizan la exactitud del clasificador. Las señales del dataset contienen filtrado RRC, ruido y canal; no tienen por qué reproducir los valores ideales de la tabla.

## 8. Procesar y guardar cada conjunto

`process_split("train")` carga `data/train/X.npy`. `mmap_mode="r"` permite leer mediante un mapa de memoria sin cargar necesariamente todo el archivo a la vez. `allow_pickle=False` evita cargar objetos serializados con pickle.

Comprueba la forma `(N, 2, 1024)` y cuenta las filas de `meta.csv`:

```python
metadata_count = sum(1 for row in csv.DictReader(file))
```

La expresión genera un uno por fila y los suma. Coincidir en longitud es necesario, pero no verifica que las etiquetas estén correctamente ordenadas.

Después reserva una matriz `(N, 6)` con `np.empty`. Esta función no inicializa todos sus valores a cero, así que el bucle debe llenar cada posición antes de guardar.

```python
iq = X[index]
signal = iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)
features[index] = extract_features(signal)
```

Cada fila de características conserva el índice de la observación original. `try/except` captura errores de validación y añade el conjunto y el índice afectados. El operador `%` permite imprimir progreso cada 10 000 observaciones.

El bucle procesa también la clase `ruido`: no contiene una condición que la excluya.

| Salida por conjunto | Contenido |
|---|---|
| `features.npy` | Matriz numérica `(N, 6)`. |
| `features_info.json` | Columnas, forma, tipo de datos y descripción del preprocesamiento. |

`json.dump` escribe la descripción; `indent=2` la hace legible y `ensure_ascii=False` conserva caracteres como tildes. El campo `noise_power_subtraction=False` documenta que no se resta una estimación de potencia de ruido.

### Opciones de ejecución

```bash
python src/features.py --check-only
```

Ejecuta únicamente las pruebas. `argparse` interpreta la opción y `action="store_true"` la convierte en un booleano.

```bash
python src/features.py
```

Ejecuta las pruebas y procesa `train`, `val` y `test`. Guarda los archivos de características y reemplaza los que ya existan con esos nombres.

```python
if __name__ == "__main__":
    main()
```

Este punto de entrada ejecuta `main` al lanzar el archivo directamente, pero no al importarlo como módulo.

---

# Parte II. Entender `04_features.ipynb`

## 9. Qué aprende el modelo

Aunque su nombre menciona features, el notebook ya realiza entrenamiento y evaluación. Cada ejemplo de entrenamiento combina:

$$
\mathbf x_i=[|C_{20}|,|C_{40}|,C_{42},|C_{60}|,C_{63}],
\qquad y_i=\text{modulación real o ruido}.
$$

El modelo aprende una regla $\widehat y_i=f(\mathbf x_i)$. No recibe el Es/N0 como característica ni las muestras I/Q directamente. Las etiquetas verdaderas de validación se usan para medir resultados, no para producir las predicciones.

## 10. Primera celda: localizar el proyecto

`Path.cwd().resolve()` obtiene la carpeta de trabajo. El código recorre esa carpeta y sus antecesoras hasta encontrar `data/train/features.npy`.

| Construcción | Significado |
|---|---|
| `(current,)` | Tupla con un elemento; la coma es necesaria. |
| `tuple(current.parents)` | Convierte las carpetas superiores en una tupla. |
| `+` | Concatena las dos secuencias. |
| `folder for folder in ... if ...` | Generador que filtra carpetas válidas. |
| `next(..., None)` | Obtiene la primera coincidencia o `None`. |
| `.is_file()` | Comprueba que exista el archivo. |

Si no encuentra características, lanza `FileNotFoundError`. Luego crea `figs/clasificacion/` con `mkdir(parents=True, exist_ok=True)` y define `save_figure` para guardar PNG y PDF.

## 11. Segunda celda: cargar y seleccionar características

`load_features(split)` carga los valores, su descripción JSON y los metadatos CSV. Comprueba dimensiones, longitudes, valores finitos y presencia de las siete clases.

```python
labels = np.array([row["mod"] for row in metadata])
esn0 = np.array([float(row["snr_db"]) for row in metadata])
```

Son comprensiones de listas: extraen un campo de cada fila. Los valores del CSV son textos; `float` convierte el nivel a número.

`set(labels)` representa los nombres únicos. Compararlo con `set(CLASSES)` verifica las categorías presentes, no su balance. La comprobación de Es/N0 finito excluye el ruido, que puede llevar `NaN` porque no tiene un nivel nominal de señal asociado.

La función devuelve cuatro resultados y el código los desempaqueta en cuatro variables. También verifica que las columnas de `train` y `val` tengan el mismo orden.

```python
selected_indices = [
    i for i, name in enumerate(columns_train)
    if name != "C21"
]
X_train = F_train[:, selected_indices]
X_val = F_val[:, selected_indices]
```

`enumerate` entrega posición y nombre. `:` selecciona todas las filas y la lista de índices selecciona las columnas deseadas. El archivo excluye `C21` por su normalización conocida; no hace una selección automática según el rendimiento en validación.

Las salidas guardadas son:

```text
Features utilizadas: ['abs_C20', 'abs_C40', 'C42', 'abs_C60', 'C63']
Train: (56000, 5)
Val: (11200, 5)
```

## 12. Tercera celda: Random Forest

Un árbol de decisión aprende comparaciones sobre las características y organiza las observaciones en ramas. Por ejemplo, `C42 <= -0.7` podría ser una pregunta de un árbol; es un ejemplo ilustrativo, no un umbral extraído de tu modelo.

Un bosque combina muchos árboles. En scikit-learn se promedian sus probabilidades de clase y se elige la clase con mayor promedio. Los árboles reciben variación mediante muestreo de observaciones con reemplazo y selección aleatoria de características candidatas.

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=MODEL_SEED,
    n_jobs=2
)
```

| Parámetro | Efecto |
|---|---|
| `n_estimators=200` | Número de árboles. |
| `max_depth=20` | Profundidad máxima de cada árbol. |
| `min_samples_leaf=2` | Mínimo de muestras de entrenamiento por hoja. |
| `max_features="sqrt"` | Configura dos candidatas por división con cinco entradas; la búsqueda puede examinar más si no encuentra una partición válida. |
| `random_state=42` | Fija la semilla de la aleatoriedad del modelo. |
| `n_jobs=2` | Permite dos trabajos en paralelo. |

Estos valores son hiperparámetros elegidos antes del entrenamiento. Los umbrales de división y las características utilizadas en cada nodo se aprenden de los datos.

### Programación orientada a objetos

`RandomForestClassifier` es una clase de Python. `model` es una instancia y `.fit()` y `.predict()` son métodos del objeto. Esta acepción de «clase» es diferente de las categorías BPSK, QPSK o ruido.

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
```

La primera línea aprende con `train`; la segunda utiliza el modelo aprendido para producir una etiqueta por fila de `val`.

### Exactitud global

$$
\mathrm{accuracy}=\frac1N\sum_{i=1}^{N}\mathbf1\{\widehat y_i=y_i\}.
$$

`y_pred == y_val` produce booleanos y `np.count_nonzero` cuenta los verdaderos. La salida del notebook fue:

$$
\frac{4061}{11200}\approx\boxed{36.26\%}.
$$

Esta métrica incluye las siete clases y todos los niveles nominales del conjunto.

## 13. Cuarta celda: exactitud frente a Es/N0

```python
mask = (y_val != "ruido") & (esn0_val == level)
accuracy = accuracy_score(y_val[mask], y_pred[mask])
```

La máscara selecciona observaciones con señal al nivel actual. `&` aplica un «y» lógico elemento por elemento; cada comparación va entre paréntesis.

La celda exige 600 observaciones por nivel. El ruido real se excluye, pero una señal predicha como ruido sigue contando como error. `append` añade cada exactitud a una lista, luego convertida en arreglo con `np.asarray`.

| Es/N0 nominal (dB) | Observaciones | Exactitud guardada |
|---:|---:|---:|
| −10 | 600 | 3.50 % |
| −8 | 600 | 4.17 % |
| −6 | 600 | 5.50 % |
| −4 | 600 | 7.00 % |
| −2 | 600 | 8.33 % |
| 0 | 600 | 10.17 % |
| 2 | 600 | 16.83 % |
| 4 | 600 | 20.67 % |
| 6 | 600 | 23.67 % |
| 8 | 600 | 32.33 % |
| 10 | 600 | 36.50 % |
| 12 | 600 | 47.17 % |
| 14 | 600 | 52.83 % |
| 16 | 600 | 61.67 % |
| 18 | 600 | 66.67 % |
| 20 | 600 | 70.50 % |

Por ejemplo, a 20 dB el 70.50 % corresponde a 423 aciertos entre 600 señales.

`plt.subplots` devuelve la figura y su área de dibujo; `ax.plot` relaciona niveles y exactitudes; `set_xticks` controla las marcas del eje. `save_figure` guarda `01_exactitud_vs_esn0_val` en PNG y PDF.

### Clasificar y detectar son preguntas diferentes

| Evaluación | Qué debe acertar |
|---|---|
| Detector de energía | Presencia de transmisión. |
| Clasificador | La categoría correcta, entre las seis modulaciones y ruido. |

Una QPSK reconocida como BPSK es un error de clasificación aunque se haya identificado que existe transmisión. Esta curva no es una curva de $P_d$.

**El notebook no llama al detector de energía.** El Random Forest decide directamente entre siete categorías, por lo que el umbral analítico del detector no fija su tasa de falsas alarmas.

## 14. Quinta celda: matriz de confusión

`confusion_matrix(y_val, y_pred, labels=list(CLASSES))` construye una matriz de siete por siete. Las filas son clases reales y las columnas clases predichas. La diagonal contiene aciertos.

```python
class_totals = cm.sum(axis=1)
cm_normalized = cm / class_totals[:, None].astype(np.float64)
```

`sum(axis=1)` suma cada fila. `[:, None]` cambia la forma de `(7,)` a `(7, 1)`, permitiendo dividir cada fila por su total mediante broadcasting.

$$
\widetilde C_{ij}=\frac{C_{ij}}{\sum_kC_{ik}}.
$$

En la salida, 751 señales de 64QAM fueron clasificadas como ruido:

$$
\frac{751}{1600}\approx46.94\%.
$$

Esto significa «de las señales reales 64QAM, esta proporción fue predicha como ruido». No significa que esa proporción de las predicciones de ruido fuera 64QAM; esa interpretación usaría otro denominador.

La diagonal normalizada es el recall de cada clase:

$$
\mathrm{recall}_i=\frac{\text{aciertos en la clase }i}{\text{observaciones reales de la clase }i}.
$$

| Clase real | Recall mostrado, aproximadamente |
|---|---:|
| BPSK | 64.6 % |
| QPSK | 26.9 % |
| 8PSK | 26.4 % |
| 16QAM | 16.5 % |
| 64QAM | 15.4 % |
| 16APSK | 25.4 % |
| Ruido | 78.5 % |

Los porcentajes de esta tabla se leen de la figura y están redondeados. El comentario numérico del notebook informa 15.44 % para 64QAM y 78.50 % para ruido.

Las matrices agregan todos los niveles de Es/N0: no demuestran que esas mismas confusiones ocurran a niveles altos. Para estudiarlo habría que filtrar las observaciones por condición.

En el dibujo, `imshow` colorea la matriz y `colorbar` añade la escala. Dos bucles anidados escriben los valores de las 49 celdas. La expresión condicional que elige texto blanco o negro solo mejora el contraste visual.

Se guardan `02_confusion_conteos_val` y `03_confusion_porcentajes_val`, ambos como PNG y PDF.

## 15. Sexta celda: redactar resultados automáticamente

```python
recall_per_class = np.diag(cm_normalized)
best_index = int(np.argmax(recall_per_class))
worst_index = int(np.argmin(recall_per_class))
```

`np.diag` extrae la diagonal. `argmax` y `argmin` entregan posiciones, mientras que `max` y `min` entregarían valores.

```python
errors = cm.copy()
np.fill_diagonal(errors, 0)
```

Se crea una copia para no modificar la matriz original y se eliminan sus aciertos. Buscar el máximo restante permite encontrar una confusión frecuente. `np.unravel_index` convierte el índice plano del máximo en fila y columna.

`.format` inserta los números y nombres en un texto; `display(Markdown(comment))` lo presenta con formato. El comentario sigue reglas escritas en el programa: no deduce automáticamente las causas físicas de los errores.

Los resultados destacados son:

- Exactitud global: **36.26 %**, 4061 de 11 200.
- Mayor recall: **ruido, 78.50 %**.
- Menor recall: **64QAM, 15.44 %**.
- Confusión destacada: **64QAM → ruido**, 751 observaciones.
- Exactitud sobre señales: **3.50 % a −10 dB** y **70.50 % a 20 dB**.

## 16. Qué podemos concluir y qué no

Las cinco características contienen información aprovechable: el modelo obtiene distintos rendimientos por clase y mejora al aumentar Es/N0 nominal. Sin embargo, la exactitud global muestra que todavía hay numerosos errores.

Muchas señales terminan en la categoría ruido. Esto es compatible con que sus características se acerquen a las del ruido en condiciones difíciles; no permite atribuir todos los errores exclusivamente al fading, al modelo o a una sola característica.

Los cumulantes calculados aquí resumen la distribución de las muestras y descartan su orden temporal. Dos señales diferentes pueden producir valores parecidos. La validación mide el efecto práctico de esa representación junto con el modelo utilizado.

El notebook usa `train` para ajustar y `val` para evaluar. **No utiliza `test`**. Aunque el extractor haya calculado sus características, eso no equivale a haber entrenado con test: la transformación se aplica por observación y no ajusta parámetros compartidos con otros conjuntos.

También guarda figuras, pero **no exporta el Random Forest entrenado**. El objeto `model` permanece en la memoria del kernel.

## 17. Conceptos reutilizables de programación

| Concepto | Ejemplo de estos archivos | Cuándo volverás a usarlo |
|---|---|---|
| Funciones | `extract_features(signal)` | Separar un cálculo reutilizable. |
| Diccionarios | `c["C42"]` | Asociar nombres con resultados. |
| Tuplas | `FEATURE_NAMES` | Definir una secuencia estable. |
| Comprensiones | `[row["mod"] for row in metadata]` | Transformar colecciones. |
| Generadores | `sum(1 for row in ...)` | Procesar elementos progresivamente. |
| Desempaquetado | `F_train, y_train, ... = load_features(...)` | Recibir varios resultados. |
| Indexación | `X[index]` | Seleccionar una observación. |
| Rebanadas | `F_train[:, selected_indices]` | Elegir filas y columnas. |
| Máscaras booleanas | `(y_val != "ruido") & (...)` | Filtrar por condiciones. |
| Broadcasting | `cm / class_totals[:, None]` | Aplicar una operación por fila. |
| Excepciones | `raise ValueError(...)` | Rechazar entradas incompatibles. |
| Gestores de contexto | `with ... as file` | Cerrar recursos automáticamente. |
| Objetos y métodos | `model.fit(...)` | Usar componentes que conservan estado. |
| Serialización | `np.save`, `json.dump` | Guardar datos y descripciones. |
| Aserciones numéricas | `assert_allclose` | Comprobar resultados esperados con tolerancias. |

## 18. Referencias y alcance

Fuentes principales: los archivos adjuntos `features.py` y `04_features.ipynb`, y las salidas conservadas dentro del notebook. Los fragmentos explicativos no reemplazan los archivos originales.

Referencias complementarias consultadas en las explicaciones:

- [RandomForestClassifier — documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- [Efficient Cumulant-Based Automatic Modulation Classification Using Machine Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10819674/).

La documentación describe la API; los valores de rendimiento de esta nota pertenecen exclusivamente a tu ejecución guardada.

---

# Apéndice. Código original de referencia

Los siguientes bloques se transcriben de los archivos adjuntos. Los del notebook dependen del orden de ejecución y de los datos del proyecto. No se ejecutaron para generar este documento. Ejecutar el procesamiento completo escribe archivos de características; ejecutar el notebook entrena el modelo y guarda figuras.


## A. features.py

```python
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
```

## B.1. Notebook: celda 1

```python
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from IPython.display import Markdown, display


current = Path.cwd().resolve()

ROOT = next(
    (
        folder
        for folder in (current,) + tuple(current.parents)
        if (folder / "data" / "train" / "features.npy").is_file()
    ),
    None
)

if ROOT is None:
    raise FileNotFoundError(
        "No se encontro data/train/features.npy. "
        "Ejecuta primero python -m src.features."
    )

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figs" / "clasificacion"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = (
    "BPSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "16APSK", "ruido"
)

ESN0_LEVELS = np.arange(-10, 22, 2)
MODEL_SEED = 42


def save_figure(fig, name):
    fig.savefig(
        FIG_DIR / (name + ".png"),
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR / (name + ".pdf"),
        bbox_inches="tight"
    )


print("Proyecto:", ROOT)
print("Figuras:", FIG_DIR)
```

Salida guardada:

```text
Proyecto: /home/lcmj2803/coms_proyecto2
Figuras: /home/lcmj2803/coms_proyecto2/figs/clasificacion
```

## B.2. Notebook: celda 2

```python
def load_features(split):
    folder = DATA_DIR / split

    features = np.load(
        str(folder / "features.npy"),
        allow_pickle=False
    )

    with (folder / "features_info.json").open(
        "r", encoding="utf-8"
    ) as file:
        info = json.load(file)

    with (folder / "meta.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        metadata = list(csv.DictReader(file))

    columns = info["columns"]

    if features.ndim != 2 or features.shape[1] != len(columns):
        raise ValueError("Forma incorrecta de features en {}.".format(split))

    if len(features) != len(metadata):
        raise ValueError("Features y etiquetas no coinciden en {}.".format(split))

    if not np.all(np.isfinite(features)):
        raise ValueError("Features no finitas en {}.".format(split))

    labels = np.array([row["mod"] for row in metadata])
    esn0 = np.array([float(row["snr_db"]) for row in metadata])

    if set(labels) != set(CLASSES):
        raise ValueError("Se esperaban las siete clases en {}.".format(split))

    if not np.all(np.isfinite(esn0[labels != "ruido"])):
        raise ValueError("Es/N0 no valido en las señales de {}.".format(split))

    return features, labels, esn0, columns


F_train, y_train, esn0_train, columns_train = load_features("train")
F_val, y_val, esn0_val, columns_val = load_features("val")

if columns_train != columns_val:
    raise ValueError("Las columnas de train y val son diferentes.")

# C21 es aproximadamente 1 por la normalizacion: no aporta discriminacion.
selected_indices = [
    i for i, name in enumerate(columns_train)
    if name != "C21"
]

selected_names = [columns_train[i] for i in selected_indices]

X_train = F_train[:, selected_indices]
X_val = F_val[:, selected_indices]

print("Features utilizadas:", selected_names)
print("Train:", X_train.shape)
print("Val:", X_val.shape)
```

Salida guardada:

```text
Features utilizadas: ['abs_C20', 'abs_C40', 'C42', 'abs_C60', 'C63']
Train: (56000, 5)
Val: (11200, 5)
```

## B.3. Notebook: celda 3

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=MODEL_SEED,
    n_jobs=2
)

print("Entrenando con train...")
model.fit(X_train, y_train)

# Una prediccion por observacion de validacion.
y_pred = model.predict(X_val)

accuracy_global = accuracy_score(y_val, y_pred)

print("Entrenamiento terminado.")
print("Exactitud global en val: {:.2%}".format(accuracy_global))
print(
    "Aciertos: {} / {}".format(
        np.count_nonzero(y_pred == y_val),
        len(y_val)
    )
)
```

Salida guardada:

```text
Entrenando con train...
Entrenamiento terminado.
Exactitud global en val: 36.26%
Aciertos: 4061 / 11200
```

## B.4. Notebook: celda 4

```python
accuracy_by_esn0 = []

print("Es/N0 (dB) | Observaciones | Exactitud")

for level in ESN0_LEVELS:
    mask = (y_val != "ruido") & (esn0_val == level)
    count = int(np.count_nonzero(mask))

    if count != 600:
        raise ValueError(
            "Se esperaban 600 señales a {} dB; hay {}.".format(
                level, count
            )
        )

    accuracy = accuracy_score(y_val[mask], y_pred[mask])
    accuracy_by_esn0.append(accuracy)

    print(
        "{:>10} | {:>13} | {:.2%}".format(
            level, count, accuracy
        )
    )

accuracy_by_esn0 = np.asarray(accuracy_by_esn0)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    ESN0_LEVELS,
    accuracy_by_esn0,
    marker="o",
    linewidth=2
)

ax.set_xlabel("Es/N0 nominal (dB)")
ax.set_ylabel("Exactitud de clasificacion")
ax.set_title("Random Forest — validacion sobre señales")
ax.set_xticks(ESN0_LEVELS)
ax.set_ylim(0.0, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
save_figure(fig, "01_exactitud_vs_esn0_val")
plt.show()
```

Salida guardada:

```text
Es/N0 (dB) | Observaciones | Exactitud
       -10 |           600 | 3.50%
        -8 |           600 | 4.17%
        -6 |           600 | 5.50%
        -4 |           600 | 7.00%
        -2 |           600 | 8.33%
         0 |           600 | 10.17%
         2 |           600 | 16.83%
         4 |           600 | 20.67%
         6 |           600 | 23.67%
         8 |           600 | 32.33%
        10 |           600 | 36.50%
        12 |           600 | 47.17%
        14 |           600 | 52.83%
        16 |           600 | 61.67%
        18 |           600 | 66.67%
        20 |           600 | 70.50%
```

## B.5. Notebook: celda 5

```python
cm = confusion_matrix(
    y_val,
    y_pred,
    labels=list(CLASSES)
)

class_totals = cm.sum(axis=1)
cm_normalized = cm / class_totals[:, None].astype(np.float64)

versions = (
    (cm, False, "02_confusion_conteos_val"),
    (cm_normalized, True, "03_confusion_porcentajes_val"),
)

for matrix, normalized, filename in versions:
    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1.0 if normalized else None
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Proporcion por clase real" if normalized else "Observaciones"
    )

    ticks = np.arange(len(CLASSES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(
        "Random Forest — matriz de confusion en val"
        + ("\nPorcentajes por fila" if normalized else "\nCantidades")
    )

    threshold = matrix.max() / 2.0

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            label = (
                "{:.1%}".format(matrix[i, j])
                if normalized
                else str(int(matrix[i, j]))
            )

            ax.text(
                j, i, label,
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()

print("Matrices guardadas en:", FIG_DIR)
```

Salida guardada:

```text
Matrices guardadas en: /home/lcmj2803/coms_proyecto2/figs/clasificacion
```

## B.6. Notebook: celda 6

```python
recall_per_class = np.diag(cm_normalized)

best_index = int(np.argmax(recall_per_class))
worst_index = int(np.argmin(recall_per_class))

errors = cm.copy()
np.fill_diagonal(errors, 0)

comment = (
    "### Comentario de los resultados\n\n"
    "El Random Forest, entrenado exclusivamente con train, obtuvo "
    "una exactitud global de **{:.2%}** sobre las siete clases de val. "
    "Se utilizaron las cinco features no constantes.\n\n"
    "La clase con mayor proporción de aciertos fue **{} ({:.2%})** "
    "y la de menor proporción fue **{} ({:.2%})**.\n\n"
).format(
    accuracy_global,
    CLASSES[best_index],
    recall_per_class[best_index],
    CLASSES[worst_index],
    recall_per_class[worst_index]
)

if errors.max() > 0:
    real_index, predicted_index = np.unravel_index(
        np.argmax(errors), errors.shape
    )

    comment += (
        "Una de las confusiones más frecuentes fue **{} → {}**, "
        "con **{} observaciones**, equivalentes al **{:.2%}** "
        "de la clase real {}.\n\n"
    ).format(
        CLASSES[real_index],
        CLASSES[predicted_index],
        errors[real_index, predicted_index],
        cm_normalized[real_index, predicted_index],
        CLASSES[real_index]
    )
else:
    comment += "No se observaron errores de clasificación en val.\n\n"

comment += (
    "La exactitud sobre señales fue **{:.2%} a −10 dB** y "
    "**{:.2%} a +20 dB**. La curva por Es/N0 excluye las "
    "observaciones de ruido puro, pero cuenta como error una señal "
    "predicha como ruido.\n\n"
    "La matriz combina todos los niveles de Es/N0; por ello, una "
    "confusión global no demuestra que dos modulaciones sean "
    "indistinguibles a Es/N0 alto. Test permanece reservado para "
    "la evaluación final."
).format(
    accuracy_by_esn0[0],
    accuracy_by_esn0[-1]
)

display(Markdown(comment))
```

Comentario guardado:

### Comentario de los resultados

El Random Forest, entrenado exclusivamente con train, obtuvo una exactitud global de **36.26%** sobre las siete clases de val. Se utilizaron las cinco features no constantes.

La clase con mayor proporción de aciertos fue **ruido (78.50%)** y la de menor proporción fue **64QAM (15.44%)**.

Una de las confusiones más frecuentes fue **64QAM → ruido**, con **751 observaciones**, equivalentes al **46.94%** de la clase real 64QAM.

La exactitud sobre señales fue **3.50% a −10 dB** y **70.50% a +20 dB**. La curva por Es/N0 excluye las observaciones de ruido puro, pero cuenta como error una señal predicha como ruido.

La matriz combina todos los niveles de Es/N0; por ello, una confusión global no demuestra que dos modulaciones sean indistinguibles a Es/N0 alto. Test permanece reservado para la evaluación final.
