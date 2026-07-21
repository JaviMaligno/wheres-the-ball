# Fase 1 — auditoría de números y conclusiones (antes de escribir)

> Recomputación independiente desde los JSON crudos (`scripts/audit_fase1.py`), con
> bootstrap, chequeo de degeneración, métrica isotrópica y exclusión de fuga. Objetivo:
> no concluir nada que los datos no sostengan. **Resultado: varias conclusiones previas
> estaban sobre-afirmadas por el tamaño de muestra (n=14 por bin).**

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
