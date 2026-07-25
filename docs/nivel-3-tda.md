# Nivel 3 · Vertiente topológica (TDA) — resultado NEGATIVO limpio

> Ejecutado 2026-07-25, ~$0 (`scripts/nivel3_tda.py`, `ripser`). **Regla fijada de
> antemano**: las features topológicas solo cuentan si BATEN a su contraparte
> geométrica (la vara del 92% de la Fase 0). No la pasan. La regla evitó teatro.

## Qué se probó

Homología persistente (Vietoris-Rips, ripser) sobre el cloud de jugadores por frame:
estadísticos globales de H0/H1 (nº de loops, persistencia máx/total, entropía de
persistencia, nacimiento del hueco más apretado) + features **localizadoras** (las
relevantes para el balón): centro del loop H1 más persistente vía su cocycle
representativo ("el balón en el agujero") y los 2 mayores círculos vacíos de Delaunay
("el balón en el hueco más grande"), cada uno con centro + radio. Mismo
`HistGradientBoostingRegressor` y mismas particiones (Metrica g1→g2, SportVU pool→eval)
que `nivel3_geometry.py`. Stride 15 (3× más grueso que el baseline; geo-vs-tda sigue
justa porque ambas usan los mismos frames — se recalcula geo-only en el mismo run).

## Resultados

| | geo-only | tda-only | geo+tda | Δ al permutar bloque TDA |
|---|---|---|---|---|
| fútbol | **0.1106** (93% del gap) | 0.2580 (−21%) | 0.1118 (≈igual) | +0.0061 (≈0) |
| basket | **0.1514** (110%) | 0.2061 (30%) | 0.1503 (ruido) | +0.0178 (≈0) |

- **tda-only ≪ geo-only**: la topología sola apenas supera al centroide (0.258 vs 0.111
  en fútbol).
- **geo+tda ≈ geo-only**: añadir el bloque topológico no mejora. Al permutar TODAS las
  columnas TDA dentro de geo+tda el error casi no se mueve → **el GBM las ignora** en
  presencia de la geometría.
- Validación de la optimización: **geo-only a stride 15 = 0.1106 ≈ 0.1115 (Fase 0,
  stride 5)** — muestrear 3× menos frames no degradó la geometría.

## Conclusión

La señal de localización del balón es **cinemática/geométrica** (dónde está la masa y
hacia dónde se mueve: convergencia de velocidades, vel-centroide), **no topológica**.
La intuición atractiva —"el balón vive en el mayor agujero de la configuración, o en el
centro del loop más persistente"— no lleva información por encima de lo que ya capturan
posición + velocidad. La homología persistente describe la *forma* del cloud a todas las
escalas, pero la posición del balón no es una propiedad de esa forma.

Resultado negativo pre-registrado: la regla del 92% hizo exactamente su trabajo.
Caveat: un featurizado por posiciones (sin ponderar por velocidad en la filtración) y
una semilla de GBM; el margen es tan grande (2× peor) que no cambia con esos detalles.

## Cierre del Nivel 3

Las tres vertientes hechas: geométrica (recupera 92%/110% — la geometría interpretable
ES casi todo el modelo profundo), informacional (MDN: calibración global débil +
**punto ciego** sobreconfiado en balones sueltos; ver `nivel-3-informacional.md`), y
topológica (no aporta). El Nivel 3 queda cerrado.
