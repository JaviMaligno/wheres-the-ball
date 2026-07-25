# Nivel 3 · Vertiente informacional — ¿cuándo es inferible el balón?

> Ejecutado 2026-07-25, ~$0 (`scripts/nivel3_informational.py`). Una Mixture Density
> Network (backbone DeepSets → cabeza de mezcla de 5 gaussianas) predice la *posterior*
> p(balón | jugadores) en vez de un punto. Fútbol, coords de campo: train Metrica g1 +
> SkillCorner (n=59167), eval Metrica g2 held-out (n=16653). Una semilla.

## Modelo

MDN con NLL. Error de punto (media de la mezcla) mediano **0.0935** — a la par del
especialista de punto del Nivel 2 (0.101 in-domain), así que la posterior no cuesta
precisión. NLL baja limpia (−1.96 → −2.57 en 20 épocas).

## 1. Calibración global (débil pero real)

Error real mediano por quintil de incertidumbre declarada (std de la mezcla):

| quintil (std↑) | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| error mediano | 0.078 | 0.089 | 0.095 | 0.103 | 0.109 |

Monótono. **corr(std declarada, error real) = +0.20.** El modelo sabe cuándo no sabe
—débilmente, pero de forma consistente y bien alimentada (n=16k).

## 2. El punto ciego (el hallazgo)

Acoplamiento = dist(balón, centroide ponderado por velocidad). Bajo = balón *trenzado*
en la masa que converge; alto = balón *suelto*, lejos.

- **[a] Error real por cuartil de acoplamiento:** Q1 (trenzado) 0.087 → Q4 (suelto)
  0.106. corr(acoplamiento, **error**) = **+0.20**, CI [0.18, 0.22]. Los balones
  sueltos son ~20% más difíciles (gap +0.009, CI [0.007, 0.011]) — son los casos duros
  de las Partes 1-2.
- **std declarada:** en esos mismos balones sueltos el modelo declara *menos*
  incertidumbre: std 0.081 (suelto) vs 0.090 (trenzado). corr(acoplamiento, **std**) =
  **−0.17**, CI [−0.19, −0.15]. No es confound de posición: dentro de bandas de
  distancia-a-portería el signo se mantiene (−0.08 / −0.23 / −0.25).

**Los dos CIs son de signo opuesto y no se solapan.** El modelo está globalmente
calibrado pero es **sistemáticamente sobreconfiado en los balones sueltos** — declara
poca incertidumbre justo donde más se equivoca. No sabe que no sabe en los casos que
importan.

**H4 salió al revés (y por eso es interesante):** predije trenzado = posterior ancha,
suelto = estrecha. Es al contrario. El trenzado es el caso *fácil* porque cuando los
jugadores convergen sobre el balón sus vectores de velocidad apuntan a él y el centroide
ponderado por velocidad se posa encima; con el balón suelto los jugadores van a
remolque y ese estimador falla. Es el mismo hilo "la velocidad es la señal" del Nivel 2,
visto desde la incertidumbre.

## 3. Mapa de inferibilidad (`wtb3-inferability-map.png`)

Error mediano por celda del campo (12×8, ≥15 muestras/celda). Lectura robusta: lo mejor
determinado son las bandas laterales y el medio campo abierto; hay una banda templada
por el corredor central y focos cálidos **frente a ambas porterías** (congestión de
área: córners, centros, rechaces donde el heurístico de convergencia se rompe). Celdas
individuales muy rojas pueden ser de n bajo; la lectura fuerte es el patrón, no la celda.

## Conclusión

La información sobre el balón no está repartida por igual: está concentrada donde los
jugadores convergen (juego trenzado, bandas, medio campo) y es escasa en balón suelto y
área. Y el modelo hereda ese gradiente **mal calibrado en el borde**: cree saber más
cuanto más suelto está el balón, que es exactamente cuando peor lo hace. Caveat: una
semilla, un partido held-out; los CIs bootstrap (n=2000) sobre el eval hacen robusto el
signo, no la generalización a otros partidos.
