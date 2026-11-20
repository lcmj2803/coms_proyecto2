# CFO, fase inicial, ganancia y multitrayecto Rayleigh

Guía para entender el canal del clasificador de modulaciones desde dos perspectivas: **comunicaciones** y **programación**.

Los ejemplos de Python son didácticos: explican las operaciones discutidas; no son una transcripción ni una revisión de la versión actual de `generator.py`.

## 1. Comunicaciones: qué le ocurre a la señal

### 1.1. Punto de partida: la señal compleja

Después de generar símbolos y aplicar el filtro RRC, tenemos muestras de banda base:

$$
x[n]=I[n]+jQ[n],\qquad n=0,\ldots,N-1.
$$

Cada número complejo contiene dos componentes del mismo instante. En el proyecto, una observación tiene $N=1024$ muestras complejas. El canal modifica esa señal antes de que llegue al detector o al clasificador.

| Efecto | Qué representa | Operación principal | Efecto visible |
|---|---|---|---|
| CFO | Diferencia de frecuencia entre transmisor y receptor | Rotación que depende del tiempo | La constelación va girando |
| Fase inicial | Desalineación inicial de fase | Rotación constante | La constelación comienza con otra orientación |
| Ganancia $A$ | Escala global de amplitud | Multiplicación por un escalar positivo | Expansión o contracción |
| Multitrayecto Rayleigh | Suma aleatoria de copias retardadas | Convolución con coeficientes complejos | Refuerzos, cancelaciones y posible interferencia entre símbolos |

### 1.2. CFO: una fase que se acumula

*Carrier Frequency Offset* es la diferencia de frecuencia entre las referencias del transmisor y del receptor. Puede deberse al desajuste de osciladores; el movimiento relativo también puede producir un desplazamiento Doppler.

$$
y[n]=x[n]e^{j2\pi f_\Delta n}.
$$

Aquí $f_\Delta$ se expresa en **ciclos por muestra**. La relación con hercios es:

$$
f_\Delta=\frac{\Delta f_{\mathrm{Hz}}}{f_s}.
$$

Cada intervalo de muestreo añade $2\pi f_\Delta$ radianes a la fase. Con el rango discutido:

$$
f_\Delta\sim U(-10^{-4},10^{-4}),
$$

el giro entre la primera y la última muestra de un fragmento de 1024 es, en el extremo positivo:

$$
\Delta\theta=360^\circ(10^{-4})(1023)\approx36.83^\circ.
$$

En 1024 intervalos sería $36.864^\circ$: de ahí la aproximación de 37 grados.

Un CFO negativo produce un giro en sentido contrario. Esta operación conserva la magnitud instantánea porque $|e^{j\theta}|=1$; en frecuencia desplaza el espectro.

### 1.3. Fase inicial: una orientación fija

$$
y[n]=x[n]e^{j\phi_0},\qquad \phi_0\sim U(0,2\pi).
$$

Todos los puntos reciben el mismo giro. Por ejemplo, en BPSK, con $\phi_0=\pi/2$:

$$
+1\longrightarrow+j,\qquad -1\longrightarrow-j.
$$

CFO y fase inicial se pueden combinar:

$$
\theta[n]=2\pi f_\Delta n+\phi_0.
$$

La fase inicial es el valor de partida; el CFO determina la pendiente de la fase respecto al índice de muestra. Si el CFO es cero, el giro permanece fijo.

### 1.4. Ganancia: amplitud y potencia no cambian por el mismo factor

Para $A>0$:

$$
y[n]=Ax[n],\qquad |y[n]|^2=A^2|x[n]|^2.
$$

| $A$ | Cambio de amplitud | Cambio de potencia | Ganancia en dB |
|---:|---|---|---:|
| 0.1 | Divide entre 10 | Divide entre 100 | −20 |
| 1 | Sin cambio | Sin cambio | 0 |
| 10 | Multiplica por 10 | Multiplica por 100 | +20 |

La expresión en dB es $20\log_{10}(A)=10\log_{10}(A^2)$.

Elegimos:

$$
u\sim U(-1,1),\qquad A=10^u.
$$

Esto produce una distribución **log-uniforme** entre 0.1 y 10. Los intervalos 0.1–1 y 1–10 tienen igual probabilidad: abarcan una década cada uno.

**La posición de la ganancia en el modelo importa.** En el modelo discutido para el dataset:

$$
y[n]=A\big(s[n]+w[n]\big),
$$

la ganancia escala señal y ruido juntos y conserva su relación de potencias. En cambio, $y[n]=As[n]+w[n]$ cambia esa relación al modificar solo la señal.

### 1.5. Multitrayecto: sumar copias con memoria

Las reflexiones hacen que lleguen contribuciones con diferentes retardos y fases. Un modelo discreto es:

$$
z[n]=\sum_{\ell=0}^{L-1}h_\ell x[n-d_\ell].
$$

Con retardos de 0, 1 y 2 muestras:

$$
z[n]=h_0x[n]+h_1x[n-1]+h_2x[n-2].
$$

Cada $h_\ell$ es complejo: modifica tanto amplitud como fase. Según sus fases, las contribuciones pueden reforzarse o cancelarse. La memoria del canal puede producir interferencia entre símbolos.

Con ocho muestras por símbolo, los retardos 1 y 2 equivalen a $1/8$ y $1/4$ de período de símbolo; **no son uno y dos símbolos completos**.

### 1.6. Qué significa Rayleigh

En este modelo, cada coeficiente se construye como una variable gaussiana compleja circular:

$$
h_\ell=\sqrt{\frac{P_\ell}{2}}(u_\ell+jv_\ell),
\qquad u_\ell,v_\ell\sim\mathcal N(0,1),
$$

con componentes independientes. Entonces:

$$
\mathbb E[|h_\ell|^2]=P_\ell.
$$

La **magnitud** $|h_\ell|$ sigue una distribución Rayleigh; el coeficiente complejo completo no es una variable real Rayleigh. Es una aproximación habitual a dispersión sin una componente determinista dominante en el trayecto modelado.

Las potencias medias relativas $(0,-10,-20)$ dB se convierten mediante:

$$
P_\ell^{\mathrm{rel}}=10^{P_{\ell,\mathrm{dB}}/10}
\quad\Rightarrow\quad(1,0.1,0.01).
$$

Estos valores son promedios estadísticos; una realización particular puede presentar otra relación entre trayectos.

Si queremos ganancia media total unitaria del perfil, podemos normalizarlos:

$$
P_\ell=\frac{P_\ell^{\mathrm{rel}}}{1+0.1+0.01}.
$$

Esta elección conserva las relaciones de potencia y hace $\sum_\ell\mathbb E[|h_\ell|^2]=1$. No obliga a que cada realización tenga exactamente esa energía.

Rayleigh no define la velocidad temporal del fading. Si sorteamos los coeficientes una vez por observación y los mantenemos fijos dentro de ella, usamos **fading por bloques**. Un perfil con trayectos secundarios débiles tampoco elimina la posibilidad de un desvanecimiento profundo.

### 1.7. Modelo conjunto y relación con el detector

Una forma del modelo discutido es:

$$
y[n]=A\left[\sqrt{\rho}\,e^{j(2\pi f_\Delta n+\phi_0)}
\sum_\ell h_\ell x[n-d_\ell]+w[n]\right].
$$

Aquí $\rho=10^{\gamma_{\mathrm{dB}}/10}$ es un factor de energía y $\mathbb E[|w[n]|^2]=1$. Interpretar $\gamma_{\mathrm{dB}}$ como $E_s/N_0$ exige una normalización coherente de símbolos y filtro; no equivale automáticamente a la SNR por muestra ni a la SNR recibida tras el fading.

El detector usa $T=P_{\mathrm{dentro}}/P_{\mathrm{fuera}}$. Una ganancia global no lo altera:

$$
T(Ay)=\frac{A^2P_{\mathrm{dentro}}}{A^2P_{\mathrm{fuera}}}=T(y).
$$

Una fase global tampoco cambia las potencias espectrales. El CFO sí puede mover energía entre las regiones, y el multitrayecto puede modificar su distribución y nivel.

## 2. Programación: cómo representar esos efectos

### 2.1. Datos y generador aleatorio

```python
import numpy as np

rng = np.random.default_rng(42)
# x: arreglo unidimensional de muestras complejas, ya generado.
n = np.arange(len(x))
```

`rng` es el generador pseudoaleatorio. Su estado avanza con cada sorteo; la misma semilla y secuencia de operaciones permiten reproducir el experimento. `n` contiene los índices 0, 1, 2, etc.; no son segundos.

### 2.2. Sortear parámetros, no cambiar la etiqueta

```python
cfo = rng.uniform(-1e-4, 1e-4)
phase = rng.uniform(0.0, 2.0 * np.pi)
amp = 10.0 ** rng.uniform(-1.0, 1.0)
```

Cada llamada produce un escalar. La modulación original continúa siendo la misma aunque el canal cambie su apariencia. Conviene registrar estos parámetros en los metadatos para analizar los errores del modelo después.

### 2.3. Aplicar CFO y fase mediante vectorización

```python
rotation = np.exp(1j * (2.0 * np.pi * cfo * n + phase))
x_rotated = x * rotation
```

| Python | Matemáticas | Significado |
|---|---|---|
| `1j` | $j$ | Unidad imaginaria |
| `2*np.pi*cfo*n + phase` | $\theta[n]$ | Un ángulo por muestra |
| `np.exp(1j * ...)` | $e^{j\theta[n]}$ | Rotación compleja de magnitud uno |
| `x * rotation` | $x[n]e^{j\theta[n]}$ | Producto elemento a elemento |

No necesitamos escribir un bucle por muestra: NumPy aplica la operación a todo el arreglo. Esto es **vectorización**.

### 2.4. Generar los coeficientes Rayleigh

```python
powers_db = np.array([0.0, -10.0, -20.0])
powers = 10.0 ** (powers_db / 10.0)
powers = powers / powers.sum()  # Elección: potencia media total unitaria.

u = rng.standard_normal(3)
v = rng.standard_normal(3)
h = np.sqrt(powers / 2.0) * (u + 1j * v)
```

`h` contiene tres coeficientes complejos. El factor `sqrt(powers / 2)` reparte la potencia media entre las partes real e imaginaria. No se normaliza después cada realización de `h`: hacerlo alteraría su distribución conjunta.

### 2.5. Aplicar el multitrayecto con convolución

```python
z = np.convolve(x, h, mode="full")[:len(x)]
```

Para retardos consecutivos 0, 1 y 2, el arreglo `h` ya representa la respuesta impulsional del canal. La convolución implementa la suma de copias retardadas.

`mode="full"` produce `len(x) + len(h) - 1` muestras; la rebanada conserva las primeras `len(x)`. Este ejemplo supone entrada cero antes del inicio y descarta la cola final. Para simular un fragmento de una transmisión continua, conviene generar contexto previo y posterior y recortar después los transitorios.

Si los retardos fueran, por ejemplo, 0, 3 y 7, habría que colocar ceros entre los coeficientes en una respuesta impulsional de longitud 8. No bastaría con convolucionar con tres coeficientes contiguos.

### 2.6. Unir las etapas en el orden del modelo

Este ejemplo presupone que `x` es una señal compleja válida y que `rng`, `cfo`, `phase`, `amp` y `h` ya se definieron:

```python
# 1. Copias retardadas y sumadas.
z = np.convolve(x, h, mode="full")[:len(x)]

# 2. Rotación del resultado del canal.
n = np.arange(len(z))
z = z * np.exp(1j * (2.0 * np.pi * cfo * n + phase))

# 3. Escala de señal y ruido complejo de potencia media unitaria.
gamma_db = 10.0  # Valor ilustrativo; requiere definir la referencia energética.
rho = 10.0 ** (gamma_db / 10.0)
w = (rng.standard_normal(len(z))
     + 1j * rng.standard_normal(len(z))) / np.sqrt(2.0)

# 4. Ganancia global sobre señal y ruido.
y = amp * (np.sqrt(rho) * z + w)
```

Aquí `z` va almacenando resultados intermedios. `y` es el resultado final. La multiplicación por `amp` ocurre después de sumar el ruido para conservar la relación señal–ruido frente a esa ganancia.

### 2.7. Comprobaciones para entender cada operación

| Experimento aislado | Resultado esperado |
|---|---|
| CFO = 0 y fase = 0 | La rotación vale 1 en todas las muestras |
| Solo fase inicial | El módulo de cada muestra no cambia |
| Solo CFO | El módulo no cambia; la fase acumula un giro |
| Ganancia = 10 | La potencia de todo el fragmento se multiplica por 100 |
| Canal `h = [1, 0, 0]` | La salida recortada de la convolución coincide con la entrada |
| Multiplicar el fragmento completo por 10 | El estadístico $T$ se conserva dentro del redondeo numérico |

Estas son predicciones del modelo para experimentos aislados, no resultados de una ejecución de tu generador.

## 3. Cómo trasladarlo a otros proyectos

El patrón reutilizable es:

1. Definir el fenómeno y su ecuación.
2. Determinar qué variables son escalares y cuáles son arreglos.
3. Elegir qué parámetros serán aleatorios y su distribución.
4. Implementar cada transformación por separado.
5. Comprobar un caso sencillo con resultado conocido.
6. Combinar las etapas y registrar sus parámetros.

En comunicaciones, esto construye una simulación del receptor. En otros proyectos, las mismas herramientas permiten modelar ganancias de sensores, desfases, retardos, ruido y respuestas de sistemas.
