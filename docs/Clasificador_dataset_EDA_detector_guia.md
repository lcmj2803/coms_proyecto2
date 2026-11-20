# Del generador BPSK al detector de energía

## Cómo leer esta guía

Esta nota reúne el resto del contenido técnico explicado en el chat, separado en **comunicaciones y análisis de señales** y **programación e implementación**. Incluye también el diseño experimental del dataset, porque conecta ambos mundos.

La explicación de CFO, fase inicial, ganancia y multitrayecto está en [CFO_fase_ganancia_Rayleigh.md](CFO_fase_ganancia_Rayleigh.md). Coloca ambos archivos en la misma carpeta para conservar el enlace.

Los fragmentos de código son explicativos: algunos proceden de los archivos compartidos y otros son ejemplos mínimos. No constituyen un único programa para ejecutar de arriba abajo. Los resultados identificados como registrados provienen de las salidas guardadas de `03_deteccion.ipynb`; esta guía no implica una nueva ejecución del dataset.

## Mapa de lo realizado

| Etapa | Pregunta que responde | Resultado |
|---|---|---|
| Entorno de Python | ¿Dónde trabajar y con qué bibliotecas? | Carpeta y entorno virtual |
| Generación | ¿Cómo representar símbolos y señales? | Muestras complejas I/Q |
| Dataset | ¿Cuántos ejemplos y con qué condiciones? | Conjuntos y metadatos |
| EDA | ¿Los datos tienen la estructura y el balance esperados? | Conteos, controles y figuras |
| Detección | ¿Hay evidencia de una señal en la banda esperada? | Estadístico, umbral y decisión |
| Clasificación posterior | ¿Qué modulación contiene una señal? | Etapa distinta del detector; no entrenada en los bloques revisados |

# Parte I. Comunicaciones y análisis de señales

## 1. Bits, símbolos, muestras y observaciones

Son cuatro niveles diferentes:

| Concepto | Significado | Ejemplo |
|---|---|---|
| Bit | Unidad binaria de información | 0 o 1 |
| Símbolo | Un punto de la constelación | $-1+j0$ en BPSK |
| Muestra | Un valor de la señal en un instante discreto | $x[n]=I[n]+jQ[n]$ |
| Observación | Fragmento completo que recibe el modelo | 1024 muestras complejas |

Una constelación de $M$ puntos transporta $\log_2(M)$ bits por símbolo cuando se usa un mapeo binario completo de longitud fija. BPSK tiene un bit por símbolo; QPSK, dos; 16QAM, cuatro.

Una muestra compleja contiene dos números reales del **mismo instante**:

$$
x[n]=I[n]+jQ[n].
$$

Por tanto:

$$
1024\text{ muestras complejas}=2048\text{ valores reales I/Q}.
$$

Esto no significa que la señal original tuviera 2048 muestras temporales.

## 2. Derivar el mapeo BPSK

Elegimos la convención:

$$
0\mapsto-1,\qquad1\mapsto+1.
$$

Buscamos una transformación afín $a_k=\alpha b_k+\beta$:

$$
\begin{aligned}
b_k=0&:\quad-1=\beta,\\
b_k=1&:\quad1=\alpha+\beta\Rightarrow\alpha=2.
\end{aligned}
$$

De ahí:

$$
\boxed{a_k=2b_k-1}.
$$

La misma correspondencia puede expresarse mediante fase:

$$
a_k=e^{j\pi(1-b_k)}.
$$

Para $b_k=0$, la fase es $\pi$; para $b_k=1$, es cero. Los símbolos tienen magnitud unitaria y una separación de fase de 180 grados.

Generar 32 símbolos no produce 32 posiciones distintas en la constelación: se superponen en los dos puntos permitidos. Cambiar la semilla cambia la secuencia, pero conserva el conjunto de puntos posibles.

## 3. Normalización de las constelaciones

La condición:

$$
\mathbb E[|a_k|^2]=1
$$

establece una energía media de símbolo de referencia. Para $M$ símbolos equiprobables:

$$
\frac1M\sum_{m=0}^{M-1}|a_m|^2=1.
$$

En QAM, distintos puntos pueden tener distinta energía: es el promedio el que vale uno. Normalizar permite comparar modulaciones sin introducir diferencias arbitrarias de escala.

## 4. Del símbolo a la señal: sobremuestreo y RRC

Con ocho muestras por símbolo, la señal conformada se escribe:

$$
x[n]=\sum_k a_k\,h_{\mathrm{RRC}}[n-8k].
$$

Cada símbolo multiplica un pulso desplazado. Las copias se suman:

$$
x[n]=a_0h[n]+a_1h[n-8]+a_2h[n-16]+\cdots.
$$

Ocho muestras por símbolo **no significa repetir ocho veces cada valor**. Los valores intermedios se obtienen mediante el filtro y la superposición de pulsos.

El RRC controla la ocupación espectral. El pulso ideal tiene duración infinita; `span=10` expresa una extensión práctica de 10 períodos de símbolo. La realización finita introduce aproximaciones.

El RRC transmisor y un RRC receptor adaptado producen idealmente una respuesta conjunta de coseno alzado que satisface el criterio de Nyquist de cero interferencia entre símbolos en los instantes apropiados. Ese criterio no es el mismo enunciado que el teorema de muestreo.

Si $R_s$ es la tasa de símbolos:

$$
f_s=8R_s,\qquad
B_{\mathrm{Hz}}=\frac{1+\beta}{2}R_s.
$$

Para $\beta=0.35$, la banda ideal normalizada ocupa:

$$
|f|\leq\frac{1.35}{16}=0.084375\text{ ciclos/muestra}.
$$

El ancho total, de extremo negativo a positivo, es $0.16875$ ciclos/muestra. No confundir ese ancho con su semiancho $0.084375$.

## 5. Muestreo, cuantificación y longitud del fragmento

| Decisión | Qué controla | Relación |
|---|---|---|
| Muestreo | Separación temporal de las mediciones | $T_s=1/f_s$ |
| Cuantificación | Niveles permitidos para representar amplitudes | $2^b$ niveles para $b$ bits en un cuantificador de longitud fija |
| Longitud del fragmento | Número de muestras conservadas | $T_{\mathrm{ventana}}=N/f_s$ |

El teorema de muestreo relaciona la frecuencia de muestreo con el contenido espectral, no la cantidad de muestras guardadas. En el caso habitual de una señal real de banda base limitada a $f_{\max}$, elegimos $f_s>2f_{\max}$ para evitar aliasing.

Si $f_s=8000$ muestras/s y $N=1024$, la ventana tiene duración convencional:

$$
T_{\mathrm{ventana}}=1024/8000=0.128\text{ s}.
$$

La separación entre la primera y la última marca temporal es $(N-1)/f_s$; el bloque representa $N$ intervalos de muestreo al hablar de duración de registro o FFT.

En el proyecto, `sps=8` hace que 1024 muestras abarquen 128 períodos de símbolo. Debido al filtro y al recorte, pueden contribuir símbolos vecinos fuera de esa ventana.

Guardar números como `float32` limita la precisión numérica, pero no equivale a haber modelado explícitamente un ADC uniforme de 32 bits con rango de voltaje definido.

## 6. Diseño del dataset

Clases: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 16APSK y ruido.

Los niveles nominales de $E_s/N_0$ son:

$$
-10,-8,\ldots,18,20\text{ dB}.
$$

Su cantidad es:

$$
\frac{20-(-10)}2+1=16.
$$

Una **pareja** es una combinación de modulación y nivel, por ejemplo `(QPSK, 10 dB)`. Generar 500 observaciones por pareja significa producir 500 fragmentos distintos bajo esa condición nominal.

| Conjunto | Por pareja | Seis modulaciones | Solo ruido | Total previsto |
|---|---:|---:|---:|---:|
| train | 500 | $6\times16\times500=48000$ | 8000 | 56000 |
| val | 100 | 9600 | 1600 | 11200 |
| test | 100 | 9600 | 1600 | 11200 |
| Total | — | 67200 | 11200 | 78400 |

La clase ruido recibe tantos ejemplos como una modulación completa. Su `snr_db` se guarda como `NaN`: los 16 niveles se usan para igualar el conteo, no para asignar energía de símbolo a un ejemplo sin transmisión.

Los papeles de los conjuntos son diferentes:

- **train:** aprender parámetros del modelo.
- **val:** elegir ajustes, arquitectura o punto de parada.
- **test:** evaluar el resultado final, sin usarlo para tomar esas decisiones.

En el diseño discutido se generan separadamente con semillas 1, 2 y 3. Una semilla distinta ayuda a separar secuencias aleatorias, pero la independencia también requiere no reutilizar transmisiones o fragmentos entre conjuntos.

## 7. Qué comprueba el EDA

### Balance por modulación y nivel

Contamos:

$$
C(m,\gamma)=\sum_i\mathbf1\{m_i=m,\ \gamma_i=\gamma\}.
$$

Para train esperamos $C(m,\gamma)=500$ en las 96 combinaciones de las seis modulaciones y los 16 niveles. Este control examina las etiquetas; no mide el $E_s/N_0$ real de las muestras.

### Duplicados

Buscamos observaciones idénticas byte por byte. Una copia de entrenamiento presente en prueba puede producir resultados de evaluación demasiado optimistas.

Cero duplicados exactos no demuestra independencia completa: dos fragmentos pueden compartir transmisión original, solaparse o diferir únicamente por ruido o ganancia.

### Inspección espectral

Una PSD muestra dónde se concentra la potencia. Una forma de banda común a BPSK, QPSK y QAM puede venir del mismo pulso y tasa de símbolos; no basta por sí sola para identificar la modulación.

## 8. FFT, frecuencias y potencia

La DFT transforma $N$ muestras temporales en $N$ coeficientes complejos:

$$
Y[k]=\sum_{n=0}^{N-1}y[n]e^{-j2\pi kn/N}.
$$

La FFT es un algoritmo eficiente para calcularla. Los coeficientes contienen información de amplitud y fase, con la escala determinada por la normalización elegida.

Con separación temporal $d$, la separación entre bins es:

$$
\Delta f=\frac1{Nd}.
$$

Para $N=1024$ y $d=1$, obtenemos 1024 frecuencias en ciclos/muestra. No se generan 1024 nuevas muestras temporales: se asigna una frecuencia a cada coeficiente espectral.

La potencia espectral usada en el detector es proporcional a:

$$
p[k]=\frac{|Y[k]|^2}{N}.
$$

Para mostrar una magnitud de potencia en decibelios usamos $10\log_{10}(p/p_{\mathrm{ref}})$. La referencia numérica uno no convierte automáticamente la gráfica en dBm.

## 9. Welch: una estimación más estable para visualizar

Welch divide la observación en segmentos, aplica una ventana, obtiene sus periodogramas y los promedia.

Para 1024 muestras, segmentos de 256 y solapamiento de 128:

$$
D=256-128=128,\qquad
L=1+\frac{1024-256}{128}=7.
$$

La ventana Hann reduce la fuga asociada a los bordes, a costa de ensanchar las componentes. El promedio reduce fluctuaciones; los segmentos solapados no son observaciones independientes completas.

| FFT completa del detector | Welch de la visualización |
|---|---|
| Una FFT de 1024 puntos | Siete FFT de 256 puntos con Hann |
| Sin ventana adicional | Ventana y promedio de periodogramas |
| Separación de bins $1/1024$ | Separación de bins $1/256$ |
| Base de la calibración F descrita | PSD más suave para inspeccionar la banda |

Welch también utiliza FFT. Ambos pueden revelar una banda similar, pero no tienen la misma distribución estadística ni producen necesariamente el mismo cociente de potencias.

## 10. Detector por razón de potencias

### Hipótesis y regiones

$$
\mathcal H_0:y[n]=Aw[n],\qquad
\mathcal H_1:y[n]=A(s[n]+w[n]).
$$

Elegimos:

$$
\mathcal B_{\mathrm{in}}=\{k:|f_k|\leq0.084375\},\qquad
\mathcal B_{\mathrm{out}}=\{k:|f_k|>0.1\}.
$$

La región intermedia es de guarda. Fuera de la banda ideal puede quedar fuga de señal por el filtro finito y el recorte; por eso hablamos de una referencia **dominada por ruido**, no de ausencia garantizada de señal bajo $\mathcal H_1$.

### Estadístico

$$
T=\frac{\frac1{K_{\mathrm{in}}}\sum_{\mathcal B_{\mathrm{in}}}p[k]}
{\frac1{K_{\mathrm{out}}}\sum_{\mathcal B_{\mathrm{out}}}p[k]}.
$$

Usamos promedios por bin porque las regiones tienen distinto tamaño. Bajo ruido blanco, $T$ suele estar cerca de uno; con señal concentrada en banda, tiende a aumentar. Es adimensional y no es directamente $E_s/N_0$.

### Invariancia frente a ganancia

Por linealidad de Fourier, $y'=cy$ implica $Y'=cY$. Para $c\neq0$:

$$
T(cy)=\frac{|c|^2P_{\mathrm{in}}}{|c|^2P_{\mathrm{out}}}=T(y).
$$

Multiplicar toda la observación por 10 multiplica ambas potencias por 100. Si se modificara solamente la señal conservando el ruido fijo, la conclusión no sería la misma.

## 11. Por qué aparece una distribución F

Bajo ruido blanco gaussiano complejo circular y una DFT unitaria, cada coeficiente tiene partes real e imaginaria gaussianas independientes de varianza $\sigma^2/2$. Entonces:

$$
\frac{2|Y_u[k]|^2}{\sigma^2}\sim\chi^2_2.
$$

La energía de un bin es exponencial; el coeficiente complejo es gaussiano. Cada bin aporta dos grados de libertad reales.

Al sumar bins de regiones disjuntas:

$$
U=\frac2{\sigma^2}\sum_{\mathrm{in}}|Y_u[k]|^2
\sim\chi^2_{2K_{\mathrm{in}}},\qquad
V=\frac2{\sigma^2}\sum_{\mathrm{out}}|Y_u[k]|^2
\sim\chi^2_{2K_{\mathrm{out}}}.
$$

Ambas sumas son independientes. Por definición de la distribución F:

$$
\boxed{T=\frac{U/(2K_{\mathrm{in}})}{V/(2K_{\mathrm{out}})}
\sim F(2K_{\mathrm{in}},2K_{\mathrm{out}})}.
$$

En el detector:

$$
K_{\mathrm{in}}=173,\quad K_{\mathrm{out}}=819,
\quad K_{\mathrm{guarda}}=32,
$$

y por tanto $T\mid\mathcal H_0\sim F(346,1638)$.

La media es ligeramente mayor que uno:

$$
\mathbb E[T]=\frac{1638}{1638-2}\approx1.001222.
$$

La derivación supone muestras de ruido independientes, circulares y gaussianas, FFT de los datos originales sin ventana adicional y regiones disjuntas. Aplicar Hann, Welch, ruido coloreado o contar bins de relleno con ceros como independientes exige revisar la calibración.

## 12. Umbral, cuantil y falsas alarmas

Definimos:

$$
P_{\mathrm{FA}}=P(T>\eta\mid\mathcal H_0).
$$

Para $P_{\mathrm{FA}}=0.01$:

$$
\eta=F^{-1}_{346,1638}(0.99)\approx1.208423.
$$

El **cuantil 0.99 es una posición en el eje horizontal**. Deja el 99 % del área a la izquierda y el 1 % a la derecha. No es la región exterior ni un valor de 0.99 para $T$.

La decisión es:

$$
T>\eta\Rightarrow\text{se detecta señal}.
$$

La línea $T=1$ es una referencia, no este umbral. Tampoco significa que el detector tenga un 99 % de exactitud general: falta evaluar la probabilidad de detección $P_D=P(T>\eta\mid\mathcal H_1)$.

En una gráfica de densidad, la altura puede superar uno. La probabilidad es el área sobre un intervalo, y el área total vale uno.

## 13. Validación con ruido independiente

Con $M=10000$ fragmentos nuevos de ruido:

$$
\widehat P_{\mathrm{FA}}=\frac{\#\{T_i>\eta\}}{M}.
$$

Si las pruebas son independientes y la calibración es correcta, el número de falsas alarmas es binomial y:

$$
\operatorname{std}(\widehat P_{\mathrm{FA}})
=\sqrt{\frac{0.01(0.99)}{10000}}\approx0.001.
$$

El rango 0.008–0.012 es una referencia aproximada de dos desviaciones estándar, no una garantía. No debe reajustarse el umbral hasta forzar que una simulación caiga dentro del rango.

El histograma del ruido de train es una comparación descriptiva. Validar con ruido nuevo y medir detección por modulación y nivel son comprobaciones posteriores distintas.

# Parte II. Programación e implementación

## 14. Carpeta, entorno y dependencias

La carpeta reúne código y resultados. El entorno virtual aísla el intérprete y los paquetes utilizados por el proyecto.

```bash
mkdir -p ~/coms_proyecto2
cd ~/coms_proyecto2
python3 -m venv .venv
source .venv/bin/activate
```

Un `requirements.txt` básico contiene:

```text
numpy
scipy
matplotlib
```

La instalación se realiza con:

```bash
python -m pip install -r requirements.txt
```

`python -m pip` ejecuta pip con ese Python; `-r` indica que debe leer una lista de requisitos. El archivo tiene que existir en la ruta indicada. Un entorno virtual no crea `requirements.txt` por sí solo.

Para comprobar las importaciones:

```bash
python -c "import numpy, scipy, matplotlib; print('Bibliotecas disponibles')"
```

Dos dificultades que aparecieron:

| Situación | Explicación | Resolución discutida |
|---|---|---|
| No se encuentra `requirements.txt` | El archivo no estaba en el directorio desde el que se ejecutó el comando | Crearlo o proporcionar su ruta correcta |
| La terminal muestra `>` tras iniciar un bloque `EOF` | El shell espera el delimitador de cierre | Cerrar correctamente el bloque o cancelar con Ctrl+C y crear el archivo de otra forma |

Una alternativa simple para crear la lista era:

```bash
printf '%s\n' numpy scipy matplotlib > requirements.txt
```

El operador `>` reemplaza el contenido de ese archivo. No se debe ejecutar sobre una lista más completa que se quiera conservar.

## 15. Programación modular y funciones

| Módulo o notebook | Responsabilidad conceptual |
|---|---|
| `generator.py` | Crear y transformar señales |
| `dataset.py` | Organizar condiciones, repeticiones, etiquetas y almacenamiento |
| Notebook de EDA | Inspeccionar datos y documentar controles |
| `detection.py` | Calcular $T$, fijar el umbral y decidir |
| `03_deteccion.ipynb` | Cargar datos y demostrar el comportamiento del detector |

Una función tiene entradas, operaciones y, opcionalmente, salidas:

```python
def aplicar_ganancia(x, amp=1.0):
    return amp * x
```

`x` y `amp` son parámetros de la definición. En `aplicar_ganancia(y, amp=10)`, `y` y `10` son argumentos de la llamada. El valor predeterminado se usa cuando no se proporciona ese argumento.

La definición no ejecuta el cuerpo. La llamada sí. La indentación delimita el cuerpo de funciones y bucles.

```python
y, info = channel(...)
```

Representa desempaquetado de resultados cuando la función devuelve dos elementos. Los puntos suspensivos de este ejemplo indican argumentos omitidos por claridad, no una llamada lista para usar.

## 16. Tipos y colecciones

| Construcción | Ejemplo | Uso |
|---|---|---|
| Entero | `index = 0` | Contadores y cantidades |
| Decimal | `amp = 0.5` | Parámetros y mediciones |
| Complejo | `1 + 2j` | Componentes I/Q |
| Texto | `mod = "QPSK"` | Nombres y etiquetas |
| Booleano | `True` | Condiciones y decisiones |
| Ausencia de valor | `None` | Resultado no encontrado o pendiente |
| Lista | `metadata = []` | Acumular registros |
| Tupla | `(0, 1, 2)` | Agrupar una secuencia inmutable |
| Diccionario | `{"train": 1}` | Asociar claves y valores |

`None` y `NaN` no son lo mismo: `None` representa ausencia de un objeto/valor, mientras que `NaN` es un valor especial de punto flotante para datos numéricos no definidos o ausentes.

```python
SEEDS = {"train": 1, "val": 2, "test": 3}
seed = SEEDS["train"]
```

Los corchetes acceden a una clave del diccionario. En un arreglo o lista, se usan también para índices; su significado depende del objeto.

Las clases del dataset son categorías. No implican que hayas definido clases de programación orientada a objetos mediante `class`.

## 17. Primer generador BPSK

```python
import numpy as np

rng = np.random.default_rng(42)
bits = rng.integers(0, 2, size=32)
symbols = (2.0 * bits - 1.0).astype(np.complex64)
```

`integers(0, 2)` incluye 0 y excluye 2: produce ceros y unos. Las operaciones sobre `bits` se aplican elemento a elemento. `astype()` convierte el tipo numérico.

`complex64` utiliza 64 bits en total: 32 para la parte real y 32 para la imaginaria. Convertir a complejo no inventa una componente Q distinta de cero; permite representarla.

Para otras tareas, el patrón es el mismo: generar o adquirir datos, transformar sus valores y elegir su representación numérica.

## 18. Bucles, condiciones y aleatoriedad

Ejemplo reducido:

```python
for mod in ["BPSK", "QPSK"]:
    for snr_db in [0, 10]:
        for repetition in range(3):
            print(mod, snr_db, repetition)
```

Produce $2\times2\times3=12$ combinaciones con repetición. `range(3)` recorre 0, 1 y 2.

`=` asigna; `==` compara; `!=` pregunta si dos valores son distintos. `and` exige que se cumplan ambas condiciones.

Se crea un generador aleatorio por conjunto y se pasa a las funciones. No se reinicia la misma semilla en cada observación, porque podría producir repeticiones.

```python
limits = (-1e-4, 1e-4)
cfo = rng.uniform(*limits)
```

El asterisco desempaqueta los dos límites como argumentos separados; equivale a `rng.uniform(-1e-4, 1e-4)`.

## 19. Arreglos, ejes y recortes

La estructura de señales es:

$$
X\in\mathbb R^{N_{\mathrm{obs}}\times2\times1024}.
$$

```python
X[index]        # Una observación: (2, 1024)
X[index, 0, :]  # I
X[index, 1, :]  # Q
```

La reconstrucción es:

```python
iq = X[index]
y = iq[0] + 1j * iq[1]
```

Ahora `y` tiene forma `(1024,)`. La coma expresa una tupla de una dimensión.

| Expresión | Selección |
|---|---|
| `x[10:20]` | Posiciones 10 a 19 |
| `x[::8]` | Posiciones 0, 8, 16, etc. |
| `x[start:start+n_samples]` | Un fragmento de longitud solicitada, si hay suficientes datos |
| `x[mask]` | Elementos donde la máscara booleana vale `True` |

`np.empty(...)` reserva un arreglo sin inicializarlo con valores útiles; hay que llenar todas las posiciones necesarias antes de guardar. `np.float32` utiliza cuatro bytes por valor real.

## 20. Metadatos y carga de archivos

`X.npy` almacena los arreglos numéricos. `meta.csv` conserva registros como modulación, nivel nominal y parámetros del canal. Debe haber una fila por observación y mantenerse el mismo orden.

```python
X_train = np.load(path, mmap_mode="r", allow_pickle=False)
```

El mapeo de memoria permite acceder al archivo en modo de solo lectura sin cargarlo completo de inmediato. `allow_pickle=False` impide cargar objetos serializados mediante pickle.

```python
with csv_path.open("r", newline="", encoding="utf-8") as file:
    metadata = list(csv.DictReader(file))
```

`with` gestiona el cierre del archivo. `DictReader` usa los encabezados como claves. Los campos CSV se leen como texto, por lo que se convierten cuando corresponde:

```python
row["snr_db"] = float(row["snr_db"])
```

La cadena `"20"` no es igual al número `20`. La cadena `"nan"` puede convertirse a `float`, produciendo un NaN.

## 21. Contar el balance y dibujarlo

```python
from collections import Counter

counts = Counter(
    (row["mod"], row["snr_db"])
    for row in metadata
    if row["mod"] != "ruido"
)
```

La expresión generadora entrega tuplas una por una. `Counter` almacena sus frecuencias. Consultar una combinación ausente devuelve cero.

```python
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for ax, mod in zip(axes.ravel(), MODS):
    values = [counts[(mod, level)] for level in ESN0_VALUES]
    ax.bar(ESN0_VALUES, values)
    ax.axhline(OBS_PER_PAIR["train"], color="red", linestyle="--")
```

- `axes.ravel()` transforma el arreglo 2×3 de paneles en una secuencia de seis.
- `zip()` empareja un panel con cada modulación; se detiene al agotarse la secuencia más corta.
- La comprensión de listas construye las alturas de las barras en el orden de los niveles.
- Compartir ejes facilita comparar las seis modulaciones.
- `ESN0_VALUES[::2]`, usado para las marcas, no elimina barras: reduce las etiquetas.

Un límite vertical fijo en 600 recortaría visualmente conteos mayores. La etiqueta literal `"Esperado: 500"` también debe cambiar si se cambia la cantidad configurada.

## 22. Duplicados mediante bytes y SHA-256

La observación `(2,1024)` en `float32` ocupa 8192 bytes. La huella SHA-256 tiene 32 bytes:

```python
fragment = X[index].tobytes(order="C")
fingerprint = hashlib.sha256(fragment).digest()
```

El orden C recorre primero una fila y después la otra. Igual contenido binario produce igual huella. Una huella coincidente es un candidato; la comparación de bytes confirma el duplicado o detecta una colisión.

```python
if fingerprint in seen:
    first_index = seen[fingerprint]
    if fragment != X[first_index].tobytes(order="C"):
        raise RuntimeError("Colision de hash detectada.")
    repeated.append((first_index, index))
else:
    seen[fingerprint] = index
```

`seen` es un diccionario de huella a primera posición. `repeated` guarda apariciones adicionales. Si A aparece en índices 0, 2 y 3, se registra `(0,2)` y `(0,3)`: dos repeticiones, no todas las parejas posibles.

Entre conjuntos:

```python
for a, b in combinations(("train", "val", "test"), 2):
    common = hash_indexes[a].keys() & hash_indexes[b].keys()
```

`combinations(...,2)` produce train/val, train/test y val/test. `&` calcula la intersección de claves. Cada huella representa un contenido distinto compartido, aunque tenga varias copias dentro de un conjunto.

El control es de bytes, no de similitud aproximada ni de etiquetas. Requiere representaciones compatibles para interpretar sus coincidencias. El bloque revisado informa resultados; no elimina duplicados automáticamente.

## 23. Rutas y exportación de figuras

```python
FIG_DIR = ROOT / "figs" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, name):
    fig.savefig(FIG_DIR / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / (name + ".pdf"), bbox_inches="tight")
    print("Figura guardada: {}".format(name))
```

`ROOT` debe ser un objeto `Path` definido previamente. Con estos objetos, `/` une partes de una ruta. Las mayúsculas indican constantes por convención, no una restricción del lenguaje.

`parents=True` crea carpetas superiores que falten. `exist_ok=True` permite que la carpeta ya exista. Definir `save_figure` no exporta nada; hay que llamarla con un objeto figura y un nombre sin extensión.

| Formato | Característica |
|---|---|
| PNG | Imagen de píxeles; `dpi` controla la resolución de exportación |
| PDF | Puede conservar líneas y texto vectoriales; también puede incluir imágenes rasterizadas |

`bbox_inches="tight"` ajusta el área al contenido. Guardar con el mismo nombre sobrescribe los archivos existentes. En el notebook de detección, el destino concreto es `figs/deteccion/`.

## 24. Elegir una observación y visualizar su PSD

```python
index = next(
    (i for i, row in enumerate(metadata)
     if row["mod"] == "QPSK" and row["snr_db"] == 20.0),
    None
)
```

`enumerate` entrega posición y registro. `next` toma la primera coincidencia. Con `None` como segundo argumento puede detectarse de forma explícita que no hubo coincidencia; sin valor predeterminado se produciría `StopIteration`.

En la visualización:

```python
f, psd = welch(y, fs=1.0, window="hann", nperseg=256,
               noverlap=128, detrend=False,
               return_onesided=False, scaling="density")
```

`fs=1` fija unidades normalizadas; `detrend=False` evita retirar la media o tendencia; el espectro bilateral mantiene frecuencias negativas y positivas.

```python
f = np.fft.fftshift(f)
psd = np.fft.fftshift(psd)
psd_db = 10*np.log10(np.maximum(psd, 1e-20))
```

`fftshift` reordena: debe aplicarse a frecuencias y valores para mantener la correspondencia. `maximum` evita el logaritmo de cero. El mínimo numérico equivale a −200 dB respecto a uno.

`axvspan` sombrea intervalos verticales; no filtra la señal. En el bloque de Welch original, la variable `outside` se calculaba pero no se utilizaba después.

## 25. Qué significan `np.fft.fft` y `fftfreq`

```python
Y = np.fft.fft(y)
f = np.fft.fftfreq(len(y), d=1.0)
```

| Parte | Significado |
|---|---|
| `np` | Alias de NumPy |
| Primer `.fft` | Submódulo de herramientas de Fourier |
| Segundo `.fft` | Función concreta de transformada unidimensional |

No se aplica la FFT dos veces. También podría escribirse `from numpy.fft import fft` y después `fft(y)`.

Con ocho puntos:

```python
np.fft.fftfreq(8, d=1)
# [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
```

El arreglo de frecuencias respeta el orden de la FFT: `f[k]` corresponde a `Y[k]`. Para obtener hercios, se usa `d=1/fs_real` en segundos.

## 26. La función real del detector revisado

```python
def calcular_T(signal):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=1.0)
    power = np.abs(spectrum)**2 / N
    inside = np.abs(frequencies) <= BAND_EDGE
    outside = np.abs(frequencies) > OUTSIDE_EDGE
    return np.mean(power[inside]) / np.mean(power[outside])
```

Recibe un vector de muestras y devuelve un escalar. No necesita `fftshift`: las frecuencias y coeficientes ya están alineados, y las máscaras seleccionan ambas mitades por su valor absoluto.

`calcular_T` por sí sola no verifica longitud, valores finitos ni regiones vacías. La función de decisión del archivo compartido agrega controles de forma y del resultado:

```python
def detectar_senal(signal):
    if np.asarray(signal).shape != (N_SAMPLES,):
        raise ValueError("Longitud incompatible con el umbral.")
    statistic = float(calcular_T(signal))
    if not np.isfinite(statistic):
        raise ValueError("El estadistico no es finito.")
    return statistic > GAMMA
```

Los mensajes se acortaron aquí por claridad. La lógica coincide con el archivo revisado. `GAMMA` es el umbral, no el $E_s/N_0$.

Una entrada real de longitud 1024 también pasa ese control de forma. La función no comprueba explícitamente que el ruido cumpla las hipótesis complejas circulares de la calibración. Un vector completamente nulo produce una razón indefinida y es rechazado por el control de finitud.

## 27. Distribuciones como objetos de Python

```python
from scipy.stats import f as f_distribution

T_NOISE_DISTRIBUTION = f_distribution(2*K_INSIDE, 2*K_OUTSIDE)
GAMMA = float(T_NOISE_DISTRIBUTION.ppf(1-P_FA))
```

`as` da un alias que evita confundir la distribución F con el arreglo de frecuencias `f`. Llamar a `f_distribution` con sus parámetros crea un objeto de distribución fijada; no genera muestras por sí solo.

| Método | Significado |
|---|---|
| `.pdf(t)` | Densidad en $t$ |
| `.cdf(t)` | Probabilidad a la izquierda de $t$ |
| `.sf(t)` | Probabilidad de superar $t$ |
| `.ppf(q)` | Cuantil que deja probabilidad $q$ a su izquierda |
| `.mean()` | Media teórica |
| `.std()` | Desviación estándar teórica |

En el notebook se verifica `.sf(GAMMA) = 0.01`. El umbral se obtiene de la distribución teórica, sin ajustarlo a train.

## 28. Comprobaciones numéricas y gráficas

```python
T_original = float(calcular_T(y))
T_amplificada = float(calcular_T(10*y))
assert np.isclose(T_original, T_amplificada, rtol=1e-6, atol=0)
```

`assert` exige una condición y produce `AssertionError` si falla. `isclose` tolera redondeo; aquí compara la diferencia con $10^{-6}|T_{\mathrm{amplificada}}|$. Para validación permanente de entradas se prefieren condiciones explícitas y excepciones, como hace el detector.

En el histograma, `density=True` normaliza áreas. `np.linspace` crea una malla para dibujar la curva y `fill_between` sombrea la cola derecha. La parte sombreada dentro de un rango finito puede omitir una pequeña porción de la cola que continúa hacia infinito.

`T_noise.std(ddof=1)` usa $M-1$ en el denominador de la varianza muestral. No es lo mismo que la desviación estándar teórica del objeto distribución.

## 29. Cómo se conecta el notebook con `detection.py`

El notebook compartido busca la raíz ascendiendo desde el directorio de trabajo hasta encontrar `data/train/X.npy` y `meta.csv`. Después busca el detector por orden en `src/`, la raíz y `notebooks/`.

```python
spec = importlib.util.spec_from_file_location("project_detection", str(DETECTION_PATH))
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)
calcular_T = detection.calcular_T
```

Esto carga el módulo desde una ruta concreta. Al ejecutarse su nivel superior, se calculan también las constantes y el umbral. La asignación final crea otro nombre para la misma función importada.

El notebook calcula algunos conteos de bins de forma explicativa; posteriormente toma el umbral y la distribución del módulo. Todos los estadísticos observados se calculan con su función `calcular_T`.

En scripts generales puede usarse:

```python
if __name__ == "__main__":
    main()
```

Este patrón ejecuta `main()` al lanzar el archivo directamente, pero no al importarlo. Es un concepto discutido para organizar programas; no es necesario para que un módulo que solo define funciones y constantes pueda importarse.

# Parte III. Evidencia, límites y transferencia

## 30. Resultados registrados en `03_deteccion.ipynb`

Estos valores proceden de las salidas guardadas del archivo compartido, no de una ejecución nueva realizada al redactar esta nota.

| Comprobación | Resultado registrado |
|---|---:|
| Forma de train | `(56000, 2, 1024)` |
| Observación seleccionada | Índice 15500, QPSK, nivel nominal 20 dB |
| $T$ de la observación | 135.883560 |
| $10\log_{10}(T)$ | 21.33 dB |
| $T$ original y amplificado por 10 | Ambos 135.88356018 en la impresión guardada |
| Bins interiores / exteriores | 173 / 819 |
| Distribución del ruido | F(346, 1638) |
| Umbral | 1.208423 |
| Probabilidad teórica de superar el umbral bajo ruido | 0.0100 |
| Detección para esa QPSK | `True` |
| Observaciones de ruido analizadas | 8000 |
| Media empírica / teórica | 1.000979 / 1.001222 |
| Desviación estándar empírica / teórica | 0.085406 / 0.083837 |

La gran diferencia entre $T$ y el umbral explica la decisión para esa observación. Las estadísticas del ruido son próximas a las teóricas. Ninguno de estos resultados demuestra por sí solo la capacidad de detectar todas las modulaciones a todos los niveles.

## 31. Qué no se debe concluir todavía

- Balance correcto no prueba que las señales estén bien moduladas.
- Ausencia de copias exactas no prueba independencia completa.
- Una banda visible no identifica QPSK de forma exclusiva.
- $T$ expresado en dB no es directamente $E_s/N_0$.
- El umbral del 1 % no significa 99 % de exactitud general.
- Una curva F que coincide visualmente no sustituye la evaluación de la cola con datos independientes.
- Las salidas revisadas no incluyen todavía la validación de 10000 ruidos nuevos ni curvas de probabilidad de detección.

## 32. Patrón que puedes reutilizar en otros proyectos

| En este proyecto | Patrón general | Otro ejemplo |
|---|---|---|
| Bits a símbolos | Traducir un modelo matemático a operaciones | Convertir voltaje a temperatura |
| Generador y dataset separados | Dividir responsabilidades | Adquisición de sensores y almacenamiento |
| Barrido modulación/nivel/repetición | Diseñar experimentos con bucles | Motor/carga/velocidad |
| Metadatos por observación | Conservar trazabilidad | Sensor, fecha y condición de medición |
| Balance y duplicados | Validar datos antes de modelar | Dataset de imágenes o vibraciones |
| $T$ y su umbral | Convertir una medida en una decisión | Detección de eventos con una referencia estadística |
| Notebook que importa funciones | Separar implementación y explicación | Experimentos de control o robótica |

Para replicarlo: define primero la entrada, la salida y la ecuación; implementa una función pequeña; comprueba un caso conocido; automatiza repeticiones; guarda datos y condiciones; valida antes de interpretar resultados.

## Fuentes técnicas consultadas durante la explicación

- [SciPy: estimación de PSD con Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
- [NIST: distribución F](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm).
- [MathWorks: filtrado de coseno alzado y aproximaciones finitas](https://www.mathworks.com/help/comm/ug/raised-cosine-filtering.html).
