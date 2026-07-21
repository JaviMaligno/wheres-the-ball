# Fase 1 — auditoría de números y conclusiones (antes de escribir)

> Recomputación independiente desde los JSON crudos (`scripts/audit_fase1.py`), con
> bootstrap, chequeo de degeneración, métrica isotrópica y exclusión de fuga. Objetivo:
> no concluir nada que los datos no sostengan.

## Resultados a n=92 (muestra escalada, single frame) — AUTORITATIVO

Tras escalar de 42 a **92 ítems** (tope de test a 2 ítems/clip; near=40, far=40, mid=12):

**Métrica primaria — correlación predicción↔GT, con IC bootstrap:**

| Modelo | corr_x [IC95%] | corr_y [IC95%] | señal |
|---|---|---|---|
| Claude Opus 4.8 | +0.37 [+0.16,+0.55] | +0.34 [+0.04,+0.59] | **sí (x e y)** |
| GPT-5.4 | +0.26 [+0.02,+0.49] | +0.17 [−0.08,+0.43] | parcial (solo x) |
| Qwen2.5-VL-7B (abierto) | +0.05 [−0.16,+0.26] | +0.21 [−0.02,+0.42] | muy débil (y borderline) |
| Claude Sonnet 4.6 | −0.01 [−0.25,+0.23] | +0.09 [−0.20,+0.37] | **no (nulo)** |

**Bin `far` (balones descentrados, n=40) — ¿se bate al baseline centro?**

| Modelo | error mediano [IC] | win-rate vs centro [IC] |
|---|---|---|
| centro | 0.357 | — |
| GPT-5.4 | 0.335 [0.18,0.48] | 55% [40,70] |
| Claude Opus 4.8 | 0.346 [0.23,0.40] | 52% [38,68] |
| Claude Sonnet 4.6 | 0.434 | 40% [25,55] |

### Conclusiones robustas (n=92)

1. **Opus 4.8 tiene la señal más clara** (corr significativa en x e y); **GPT-5.4 parcial**
   (solo x); **Sonnet 4.6 nulo**. El orden previo "GPT>Opus" era artefacto de n=42 (la
   corr_y de GPT cayó de 0.43 a 0.17 al escalar). Orden real: **Opus ≳ GPT > Sonnet(nulo)**.
   El **VLM abierto Qwen2.5-VL-7B** queda al nivel de Sonnet (señal muy débil: x nula,
   y borderline) — la inferencia del balón emerge solo en los flagship cerrados grandes.
2. **Ningún modelo bate de forma fiable al centro en balones descentrados** desde una sola
   imagen (win-rates 55/52% con IC que incluyen 50%). La intuición es **real pero
   limitada**: correlacionan con la posición del balón, pero un frame no basta para
   resolver los descentrados. (El "GPT 64% en far" de n=42 era ruido.)
3. Centroide de jugadores (B1): el peor.

## Lección del n=42 (por qué escalamos)

Con n=14/bin, los cortes por distancia al centro y RQ2 **no eran concluyentes** (IC
bootstrap amplísimos; la "casi-mitad del error en far" de RQ2 era artefacto de medianas
de grupo, movido por 2 ítems). Escalar a n=92 corrigió el ranking de modelos y descartó
el "bate al centro en far". Moraleja: reportar **correlación-con-IC** como primaria y
mejoras **pareadas** con IC, nunca medianas de grupo a n pequeño.

## Qué NO sostienen los datos (n=42; 14 por bin)

- **Cortes por bin de distancia al centro no son significativos.** Bootstrap del error
  mediano en `far`: GPT 0.219 **[0.086, 0.561]**, centro 0.359 [0.331, 0.414],
  Opus 0.311 [0.074, 0.373], Sonnet 0.410 [0.310, 0.609]. Los IC se solapan; no se puede
  afirmar que GPT bata al centro en `far`, ni el orden GPT>Opus>Sonnet.
- **Win-rate pareado en `far` tampoco es concluyente.** GPT 64% **[36%, 86%]** (incluye
  50%), Opus 46% [23%, 69%], Sonnet 43% [14%, 71%].
- **RQ2 temporal: la "casi-mitad del error" es un artefacto.** Era una comparación de
  medianas de grupo (single 0.219 vs temporal 0.124), pero la mejora **pareada** en `far`
  tiene mediana Δ≈0 con IC que incluye 0 para los tres modelos. El temporal ayuda mucho
  en 2 ítems de 14 (p.ej. 0.68→0.07) y nada en el resto. No se puede concluir que el
  contexto temporal desbloquee los balones descentrados con estos datos.

## Qué SÍ sostienen los datos (métrica bien alimentada, n≈39-42)

- **Correlación de la predicción con el GT** (mucho mejor potencia que los bins):

  | Modelo | corr(x, gt_x) | corr(y, gt_y) | % pred cerca del centro |
  |---|---|---|---|
  | GPT-5.4 | +0.17 | +0.43 | 7% |
  | Claude Opus 4.8 | +0.35 | +0.07 | 5% |
  | Claude Sonnet 4.6 | −0.11 | +0.06 | 5% |

  GPT y Opus **correlacionan** con la posición real del balón; **Sonnet ~0 (no infiere)**.
  Ningún modelo es degenerado (no se limitan a predecir el centro).
- **Robustez de la métrica:** con distancia isotrópica en píxeles el orden no cambia.
- **Fuga:** el tercio `far` no contiene ningún ítem marcado por fuga → esa preocupación
  no afecta a la comparación clave.

## Conclusiones revisadas (lo que podemos decir hoy)

1. **Sonnet 4.6 no infiere el balón** (corr ~0). Robusto.
2. **GPT-5.4 y Opus 4.8 sí capturan señal** (corr con GT). Robusto en dirección; la
   magnitud y el ranking fino entre ellos **no** está resuelto.
3. Todo lo que dependa de los bins (¿bate al centro en `far`? ¿ranking en descentrados?)
   y de RQ2 (¿ayuda el temporal, y dónde?) **requiere más datos** para concluir.

## Análisis más profundo necesario (antes de publicar)

- **Escalar la muestra** (≥150-200, idealmente ~500) para estrechar los IC y resolver
  los bins y RQ2. Es el bloqueante para las conclusiones a nivel de bin/temporal.
- Reportar **correlación con IC** como métrica primaria (mejor potencia que la mediana de
  distancia con este n).
- Para RQ2, reportar la mejora **pareada** (no medianas de grupo) y su IC.
- Tests pareados (Wilcoxon) entre sistemas con corrección por comparaciones múltiples.
