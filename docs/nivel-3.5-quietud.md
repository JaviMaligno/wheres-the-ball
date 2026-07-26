# Nivel 3.5 · El modelo necesita movimiento — la quietud es su punto ciego

> Ejecutado 2026-07-27, ~$0 (`scripts/nivel35_premise.py`, `scripts/nivel35_stillness.py`).
> Semilla del artículo 4. Idea de Javier: el error amplio quizá vive en valores parados;
> explorar si lo determinable en vuelo es la dirección más que la posición. La premisa
> ingenua se cayó y surgió una mejor. Fútbol, coords de campo, Metrica g1↔g2 (robustez
> en ambas direcciones).

## Paso 0 (premisa) — se cae, y sale algo mejor

Categorizando frames por evento (SET PIECE / balón largo / pase corto / juego abierto):
el error alto NO se concentra en valores parados (el evento SET PIECE de Metrica es
instantáneo, ~71 frames) sino en juego abierto. Los balones largos SÍ son algo más
difíciles (0.134 vs 0.106 pase corto, sobre-representados 1.4× en el peor 25%), pero
modestos.

El corte revelador es por **velocidad del balón**, no por etiqueta de evento:

| estado del balón | error mediano | p90 |
|---|---|---|
| **asentado (<2 m/s)** | **0.142** | **0.314** |
| lento (2-6) | 0.100 | 0.229 |
| moviéndose (6-12) | 0.105 | 0.215 |
| vuelo (>12) | 0.121 | 0.254 |

corr(velocidad balón, error) ≈ 0. El balón *en vuelo* no es el caso duro; el **balón
inmóvil** sí. Mecanismo: el modelo vive de la velocidad ("hacia dónde va la masa que
corre"); un balón quieto no ofrece señal de movimiento.

## Experimento principal (robusto en g1↔g2)

### Parte A — el balón inmóvil

| | balón quieto | balón movido |
|---|---|---|
| error full (g1→g2 / g2→g1) | 0.142 / 0.162 | 0.105 / 0.108 |
| cuánto ayuda la velocidad | **+0.018 / +0.002** | **+0.039 / +0.038** |

La contribución de la velocidad **se desploma** (hasta ~0 en g2→g1) cuando el balón
está quieto, y es grande cuando se mueve → confirma el mecanismo. Pero la quietud es
dura por una segunda razón: **los peores balones quietos están lejos de la masa**
(acoplamiento del peor-25% 0.29-0.34 vs 0.23 global) y son **centrales**, no en banda
(dist a banda 0.41-0.42 vs 0.23-0.30). **corr(acoplamiento, error | quieto) = +0.53 /
+0.65** (robusto). Un balón muerto en medio del campo, con los jugadores repartidos y
sin converger, no tiene ninguna pista que lo delate — es el punto ciego del Nivel 3 en
su forma más pura.

### Parte B — dirección en vuelo (positiva pero modesta)

Prediciendo el vector unitario de velocidad del balón desde los jugadores, solo en
frames de vuelo (>12 m/s):

- error angular mediano **44.8° / 45.2°** (vs azar 90°, baseline media-constante ~92°).
- **~50% de los balones en vuelo con la dirección acotada a <45°.**

La dirección **es legible** de los jugadores muy por encima del azar — las carreras
anticipan por dónde va el balón. Pero **no se determina claramente mejor que la
posición** (posición en vuelo 0.120-0.126, recupera ~similar sobre su baseline); son
unidades distintas y el cruce no es limpio. Queda como "la dirección es legible", no
como "la dirección gana a la posición".

## Conclusión

El hilo de toda la escalera —la señal es el movimiento— tiene su corolario: **donde no
hay movimiento (balón inmóvil, sobre todo lejos de la masa y en el centro), el modelo
se queda ciego.** Y cuando el balón vuela rápido, aunque su posición instantánea sea
difícil, su *dirección* de viaje sí se lee en la forma en que los jugadores se mueven.
Caveat: robusto en g1↔g2 (un dataset, dos partidos); la generalización a otros partidos
/ deportes queda pendiente.
