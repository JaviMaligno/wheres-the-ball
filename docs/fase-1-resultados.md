# Fase 1 — primeros resultados (SoccerNet-Tracking + LaMa)

> Primera corrida real del harness de Nivel 1: 42 ítems de SoccerNet-Tracking (split
> test), enmascarado con **LaMa**, control de fuga, prompt `v0-neutral`, condición de
> **frame único**. Reproducible: `fase1_build_dataset.py` → `inpaint_lama.py` →
> `fase1_run.py` → `fase1_report.py`.

## Configuración

- **Dataset**: 42 frames estratificados por estado del balón (possession 12,
  short_pass 12, long_pass 12, contested 6 — la disputa pura es escasa), 1 frame por
  clip como máximo para diversidad.
- **Enmascarado**: LaMa (deep inpainting). Borra el balón sin dejar rastro sobre césped,
  líneas y jugadores (donde Telea sí delataba la posición).
- **Sistemas**: `center` (0.5,0.5), `centroid` (centroide de jugadores, B1),
  `gpt` (`gpt-5.4`, Azure), `claude` (`claude-sonnet-4-6`).

## Resultados (error = distancia euclídea normalizada; menor = mejor)

| Sistema | mediana | media | PCK@.10 | poseedor |
|---|---|---|---|---|
| center | 0.199 | 0.210 | 0.17 | 0.19 |
| centroid (B1) | 0.213 | 0.254 | 0.17 | 0.17 |
| **gpt-5.4** | **0.151** | 0.222 | **0.40** | **0.31** |
| claude-sonnet-4-6 | 0.196 | 0.254 | 0.24 | 0.19 |

(Subconjunto sin los 6 ítems marcados por fuga: center 0.199, gpt 0.168, claude 0.196,
centroid 0.201 — misma lectura.)

### Por estrato (mediana)

| Sistema | possession | short_pass | long_pass | contested |
|---|---|---|---|---|
| center | 0.266 | 0.196 | 0.195 | 0.076 |
| gpt | 0.256 | 0.157 | 0.128 | 0.035 |
| claude | 0.353 | 0.162 | 0.187 | 0.083 |

## Lectura

1. **GPT-5.4 es el único que bate con claridad el baseline "centro"** (0.151 vs 0.199) y
   domina en PCK@.10 (0.40 vs 0.17) y en acierto de poseedor (0.31 vs 0.19). Confirma
   H1 en un dataset con cámara realista y balón diminuto.
2. **Claude Sonnet 4.6 ≈ baseline centro** (0.196 vs 0.199). En esta cámara (más alta y
   abierta que el broadcast de Fase 0) Claude apenas aporta sobre el baseline trivial.
   Caveat de gama: `gpt-5.4` es flagship y Sonnet gama media → falta `claude-opus-4-8`.
3. **El centroide de jugadores (B1) es PEOR que el centro.** El baseline geométrico naíf
   no sirve: el balón no está en el centroide del equipo. (Adelanta que la geometría útil
   del Nivel 3 tendrá que ser más fina: Voronoi, pitch control, velocidades.)
4. **Estratos**: `possession` es el más difícil para todos (juego de construcción, balón
   lejos del centro y disperso); `contested` el más fácil (acción central congestionada,
   cámara centrada). Esto **refina H4**: el determinante no es posesión-vs-pase, sino
   *central-congestionado vs. amplio-disperso*.

## Caveats y arreglos para el dataset formal

- **Sesgo de cámara (confound serio).** El baseline "centro" es fuerte porque la cámara
  principal sigue el balón. Hay que reponderar/seleccionar ítems con balón descentrado, o
  el ranking queda confundido. Es la limitación nº1 del diseño y aquí se ve cuantificada.
- **Fuga en balón con motion blur.** El control de fuga marcó **6/42**; al menos uno es
  un residuo real: en balones rápidos la estela supera el bbox ajustado del GT y LaMa no
  la cubre. Arreglo: **dilatar la máscara según la velocidad** del balón o excluir frames
  con blur. Los balones lentos/estáticos quedan perfectos.
- **Falta la condición temporal (RQ2)** y `claude-opus-4-8` + un VLM abierto (Qwen-VL).
- Escala: 42 ítems → subir a ~500 estratificados con IC por bootstrap (§7 del diseño).
