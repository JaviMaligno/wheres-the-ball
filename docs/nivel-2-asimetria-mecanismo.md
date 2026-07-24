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
