# Nivel 3 · Fase 0 — la inferencia del balón ES (casi toda) geometría

> Ejecutado 2026-07-24, ~$0. Reproducible: `scripts/nivel3_geometry.py` +
> `src/wheres_the_ball/features/geometry.py`. Vertiente explicativa del diseño del
> Nivel 3: ¿cuánto del modelo deep recuperan ~10 features geométricos interpretables?

## Diseño

Por frame (formato del Nivel 2): ~10 features geométricos 2D legibles — centroide,
centroide×velocidad, jugador más rápido, jugador más denso, **punto de convergencia
del campo de velocidades** (mínimos cuadrados sobre las semirrectas de movimiento),
**contacto entre equipos** (arista Delaunay más corta entre rivales), centroides por
equipo, dispersión, velocidades medias/máx. Encima, gradient boosting (300 iters).
Mismos splits que el Nivel 2.

## Resultados

| | geo-GBM | deep (N2) | centroide | recovery del gap |
|---|---|---|---|---|
| fútbol (Metrica g2) | 0.1115 | 0.101 | 0.231 | **92%** |
| baloncesto (2 partidos) | 0.1494 | 0.158 | 0.227 | **112%** |

**Zero-shot del modelo geométrico** (cruce de deportes): fútbol→basket **0.221** (deep:
0.347) · basket→fútbol 0.198 (deep: 0.174). Transfiere **más simétricamente** que el
deep — coherente con el Nivel 2: lo que no transfería era la calibración de velocidades
específica del deporte, y el featurizado geométrico normaliza gran parte.

**Importancia por permutación (fútbol, Δ error mediano):**

| feature | Δ | |
|---|---|---|
| vel_centroid | **+0.203** | domina con un orden de magnitud |
| fastest | +0.034 | |
| spread | +0.019 | |
| home_centroid | +0.011 | |
| densest | +0.009 | |
| team_contact (Delaunay) | +0.007 | |
| converge (campo vectorial) | +0.006 | el "elegante" apenas aporta |
| resto | ≤+0.006 | |

## Lecturas

1. **La vertiente explicativa del Nivel 3 se confirma**: ~10 features interpretables
   recuperan el 92% del gap deep-sobre-centroide en fútbol y igualan/superan al deep en
   baloncesto. En primera aproximación, *inferir el balón = "seguir a la masa que
   corre"* (centroide ponderado por velocidad) + correcciones menores.
2. **El modelo interpretable transfiere mejor que el deep** entre deportes — la
   featurización geométrica elimina la calibración específica del deporte que el
   Nivel 2 identificó como cuello de botella del transfer.
3. El feature teóricamente bonito (convergencia de semirrectas) aporta poco con solo
   velocidades de movimiento; su versión fuerte necesitaría orientación corporal (no
   disponible en tracking).

## Cautelas

- "112%" = iguala/supera a *este* baseline deep (DeepSets pequeño, 1.5-2k pasos), no a
  cualquier deep. Con un deep mayor el gap podría reabrirse.
- Importancia por permutación con features correlacionados reparte crédito de forma
  imperfecta; la dominancia de vel_centroid es de orden de magnitud, robusta a eso.
- Los deep refs vienen de presupuestos de entreno ligeramente distintos (Fase 0).

## Siguiente (Nivel 3, resto)

- **TDA** con la regla autoimpuesta: cada feature topológico debe batir a su
  contraparte geométrica en ablación o se cae. La vara ahora es alta (92%).
- **Vertiente informacional**: p(balón|configuración) con MDN; entropía por estrato
  (posesión vs pase largo); mapa de inferibilidad del campo.
- Formalización del caso idealizado si el resultado lo merece.
