# Nivel 2 — auditoría de números (previa al artículo)

> Recomputación independiente y controles extra (`scripts/audit_nivel2.py`),
> 2026-07-24. A diferencia del Nivel 1 (dos correcciones), **aquí todas las
> afirmaciones aguantan**; se añade un matiz al techo.

## A. Ventaja del init real a 30 min — SIGNIFICATIVA

Pareado por semilla, juntando las tres réplicas independientes (v1 forward,
consolidación, inversa; el control warm-start original se excluye del pooling por
compartir chunks/semillas con v1):

| Contraste | semillas | victorias | Δ medio [IC95%] | sign-test |
|---|---|---|---|---|
| init real vs scratch | 13 | **12** | +0.0127 [+0.0076,+0.0179] | **p=0.002** |
| init real vs permutado | 10 | **9** | +0.0121 [+0.0068,+0.0173] | **p=0.011** |

## B. Techo imagen-space — aguanta, con matiz

- Sin sesgo de descarte: los 34 ítems far conservados son ligeramente MÁS difíciles
  (err centro 0.360) que los 6 descartados (0.342).
- **Estadística decisiva**: bate al centro en far **28/34 = 82%±13** (VLMs: 53%).
- **Matiz**: head-to-head ítem a ítem, especialista vs GPT 20/34 (59%±17) y vs Opus
  21/34 (62%±16) — favorable pero el IC toca 50% a n=34. La frase defendible es
  *"el especialista resuelve el sesgo de cámara (82%) y los VLMs no (53%)"*, no
  "gana a los VLMs ítem a ítem".

## C. Anti-correlación geométrica en imagen — robusta

Centroide corr_x = **−0.58 [−0.70, −0.44]** (bootstrap 10k; todo el IC negativo).

## D. Asimetría del transfer — NO es artefacto de volumen (control nuevo)

La fuente basket tenía 2× muestras (155k vs 76k). Submuestreada a 76k y re-entrenada:
zero-shot basket→fútbol = **0.174** (con 155k: 0.177; la dirección contraria: 0.347).
La asimetría sobrevive intacta a igualdad de datos.

## Veredicto

Los cuatro titulares del Nivel 2 quedan validados:
1. Geometría posicional anti-informativa en broadcast (cámara).
2. Transfer genuino pequeño del init real, solo a ~30 min, sobrevive al control
   permutado (p=0.011) — y la ablación lo localiza en el *uso de velocidades*.
3. Transfer asimétrico basket→fútbol (no volumen).
4. Techo de tracks resuelve el sesgo de cámara donde los VLMs están al azar.

**Listo para el artículo de blog.** (Paper de workshop: on hold hasta el Nivel 3.)
