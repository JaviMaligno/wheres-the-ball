# Nivel 2 — cierre: matriz inversa y techo especialista

> Ejecutado 2026-07-24, ~$0. Reproducible: `scripts/nivel2_reverse.py` y
> `scripts/nivel1_ceiling.py`. Con esto quedan cubiertos todos los pendientes
> experimentales del Nivel 2 (y el techo pendiente del Nivel 1).

## 1. Matriz inversa (baloncesto → fútbol): el transfer es ASIMÉTRICO

Fuente = 4 partidos SportVU (155k muestras); eval = Metrica g2; pool few-shot fútbol.

| | zero-shot | 1 min | 5 min | 30 min |
|---|---|---|---|---|
| init basket | **0.177** (corr_y +0.87) | 0.202 | 0.145 | **0.114** |
| init permutado | — | 0.244 | 0.138 | 0.129 |
| scratch | — | 0.183 | 0.141 | 0.127 |

- **Basket→fútbol zero-shot funciona**: 0.177 bate a los baselines geométricos del
  fútbol (centroide 0.231, vel-centroide 0.204) y se acerca al in-domain (0.103-0.126).
  En dirección contraria (fútbol→basket) el zero-shot era 0.347, peor que el centroide.
- En ratios zero-shot/in-domain (descuenta la dificultad del eval): ~1.7× vs ~2.2× —
  la asimetría persiste. Interpretación: el baloncesto, con el balón siempre acoplado a
  la masa, enseña un acoplamiento fuerte que generaliza; el fútbol (balón suelto,
  pelotazos) enseña uno débil.
- Few-shot: mismo patrón que la ida — la ventaja del init real solo emerge a 30 min
  (0.114 < 0.127-0.129); presupuestos bajos, ruido.

## 2. Techo especialista en espacio de imagen (cierra el Nivel 1)

DeepSets de trayectorias entrenado en el split TRAIN de SoccerNet (57 clips, 16k
muestras, **solo tracks GT** — sin imágenes), evaluado sobre los ítems del benchmark del
Nivel 1. Comparación **pareada** (mismos 75 ítems con 1 s de historia; far n=34):

| Sistema | mediana | corr_x | corr_y | far med | far win vs centro |
|---|---|---|---|---|---|
| **especialista (tracks)** | 0.195 | **+0.62** | **+0.49** | **0.227** | **28/34 (82%)** |
| GPT-5.4 | **0.148** | +0.27 | +0.16 | 0.353 | 18/34 (53%) |
| Claude Opus 4.8 | **0.147** | +0.37 | +0.34 | 0.346 | 18/34 (53%) |
| Claude Sonnet 4.6 | 0.234 | +0.01 | +0.01 | 0.439 | 13/34 |
| centro | 0.205 | — | — | 0.360 | — |

**La frase que resume el proyecto:** los tracks de los jugadores contienen información
suficiente para resolver los balones descentrados (82% de victorias vs el centro, donde
los VLMs se quedan en ~53%, el azar), y los VLMs apenas extraen nada de ella desde
píxeles. (Matiz de la auditoría: el head-to-head ítem a ítem vs GPT/Opus es 59-62%,
favorable pero no significativo a n=34 — la afirmación fuerte es contra el sesgo de
cámara, no contra los VLMs uno a uno; ver `nivel-2-auditoria.md`.) Su mejor
mediana global (0.147 vs 0.195) viene de precisión pixel en los ítems fáciles centrados
(el prior de la cámara), no de leer a los jugadores.

### Caveats

- Es un **techo de información con tracking perfecto** (usa GT de jugadores, como los
  baselines B del diseño): mide señal disponible, no percepción.
- n=75 de 92 (los ítems sin 1 s de historia completa caen); far n=34. Entreno pequeño
  (16k muestras) — el techo real con más datos sería aún más alto.

## Estado final del Nivel 2 (hipótesis del diseño original)

| Hipótesis | Veredicto |
|---|---|
| H1: transferencia zero-shot sustancial | **Refutada** en fútbol→basket; **matizada** al revés (basket→fútbol sí, asimétrico) |
| H2: velocidades > posiciones; lo más transferible | **Refutada** como señal (posiciones ≫); irónicamente el *uso* de velocidades es la parte transferible del aprendizaje |
| H3: few-shot recupera el gap | Parcial: recuperan los datos del destino; el init real añade ~4-10% solo a 30 min |

Material listo para el paper de workshop: matriz completa 2×2, tres controles
(permutado, ablación, pareado), asimetría de transfer y techo vs VLMs.

## Post-scriptum (2026-07-25): el cuadro {entrenado × input} completo

Instado por la revisión de Javier, se completaron las celdas que faltaban
(`nivel1_vlm_tracks.py` y `modal_app/cnn_pixels.py` — ResNet-18 sobre 7k frames
enmascarados con LaMa, entrenada en Modal). Far win-rate vs centro (n=34 pareado):

| | tracks | píxeles |
|---|---|---|
| **entrenado** | red diminuta **82%** (corr +0.62/+0.49) | CNN pequeña **74%** (corr +0.45/+0.49) |
| **zero-shot frontera** | GPT 35% · Opus 38% (corr ~0/negativa) | GPT 53% · Opus 53% |

**Conclusión final del cuadro: el entrenamiento importa, el input apenas.** Ni la
información falta (columna tracks) ni la percepción es el muro (celda CNN-píxeles);
los generalistas zero-shot fallan con ambos inputs — y peor con coordenadas (caen en
la trampa del centroide). El conocimiento general del juego no sustituye a la
inferencia específica de la tarea.
