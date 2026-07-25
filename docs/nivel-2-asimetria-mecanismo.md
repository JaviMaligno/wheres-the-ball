# Nivel 2 · Mecanismo de la asimetría del transfer (basket→fútbol ≫ fútbol→basket)

> Ejecutado 2026-07-24, ~$0 (`scripts/nivel2_asymmetry_mechanism.py`). Investigación
> abierta por la revisión de Javier: la asimetría (zero-shot 0.177 vs 0.347) merecía
> mecanismo, no solo interpretación.

## Tests y resultados

**1. ¿Es el basket "más acoplado"? NO (premisa ingenua falsa).** Distribución de
dist(balón, vel-centroide) en unidades normalizadas de campo:

| deporte | p25 | p50 | p75 | p90 |
|---|---|---|---|---|
| fútbol | 0.132 | 0.204 | 0.264 | 0.318 |
| basket | 0.140 | 0.219 | 0.299 | 0.374 |

Prácticamente idénticas (si acaso, basket ligeramente MÁS suelto en fracciones de
cancha). La hipótesis "el balón de basket vive pegado a la masa" no sobrevive a la
medición en el espacio en que el modelo aprende.

**2. Error zero-shot por cuartil de acoplamiento del frame destino:**

| dirección | Q1 (pegado) | Q2 | Q3 | Q4 (suelto) |
|---|---|---|---|---|
| basket→fútbol | 0.161 | 0.164 | 0.179 | 0.252 |
| fútbol→basket | 0.263 | 0.304 | 0.360 | 0.448 |

Ambas direcciones se degradan con la soltura (esperable), pero fútbol→basket es peor
**en todos los cuartiles** — el fallo no se concentra solo en balones sueltos.

**3. Test causal (PASA): entrenar fútbol solo con frames acoplados mejora el transfer.**
Con nº de muestras igualado (n=8825):

- entrenado solo en frames ACOPLADOS (≤ mediana): zero-shot basket **0.285**
- entrenado en frames mezclados: 0.334

**~15% de mejora solo filtrando el entrenamiento.** Los frames de balón suelto del
fútbol enseñan activamente el hábito de predecir lejos de la masa, y el baloncesto lo
castiga.

## Conclusión (calibrada)

La hipótesis del acoplamiento sobrevive **en versión causal-de-entrenamiento**: no es
que el basket *sea* más acoplado (no lo es, medido), sino que **la dieta de balones
sueltos del fútbol degrada el prior transferible**. Explica parte de la asimetría
(0.334→0.285 con el filtro; el basket→fútbol real está en 0.177) — **el resto queda
abierto** (candidatos: estructura de media cancha, distribución de velocidades,
patrones de ocupación). Pregunta abierta marcada para el paper de workshop.

## Cierre (2026-07-25): la asimetría ES un desajuste de escala de velocidades

`scripts/nivel2_asymmetry_decompose.py`. Test decisivo: entrenar modelos
**solo-posiciones** en ambos deportes y cruzarlos, vs features completas.

**Distribuciones de velocidad (magnitud, fracciones de campo/s):** basket ~3-4× MÁS
rápido que fútbol (basket p50 0.068 / p99 0.244; fútbol p50 0.021 / p99 0.076). Causa:
la cancha de basket (28×15 m) es diminuta frente al campo (105×68 m), así que la misma
velocidad absoluta es una fracción mucho mayor. (Corrige la intuición ingenua de
"pelotazos del fútbol = velocidades extremas": falso en el espacio normalizado.)

**Asimetría zero-shot (fútbol→basket menos basket→fútbol):**

| variante | s2b | b2s | asim. absoluta | asim. en ratio/in-domain |
|---|---|---|---|---|
| full (21-dim) | 0.347 | 0.185 | +0.162 | **+0.36** |
| solo-posiciones (11-dim) | 0.282 | 0.191 | +0.091 | **−0.11** |

En absoluto, quitar velocidades reduce la asimetría casi a la mitad. **En ratio
(descontando que el fútbol es un objetivo más fácil: in-domain 0.101 vs 0.158), la
asimetría de solo-posiciones se DESVANECE (−0.11)** → las posiciones transfieren
simétricamente en ambas direcciones; **toda la asimetría genuina vive en el canal de
velocidad.**

**Conclusión (cierra la pregunta):** la asimetría del transfer es un **desajuste de
escala de velocidades**. El uso de la velocidad (lo transferible-pero-frágil de la
ablación RQ2) se calibra a la escala del deporte de entrenamiento y falla al cruzar el
salto ~3-4×; las posiciones, sin escala propia, cruzan sin problema. El resultado
coupled-only (+15%) encaja: los frames de balón suelto del fútbol son los de mayor
velocidad. Caveat: una semilla por variante; las distribuciones de velocidad son
robustas (n enorme) y el giro en ratio (+0.36 → −0.11) es grande y limpio.
