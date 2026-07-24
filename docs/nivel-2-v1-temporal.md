# Nivel 2 · v1 temporal — lo que transfiere es la dinámica (y solo como prior)

> ⚠️ **Corregido por la ablación RQ2 ([`nivel-2-rq2-ablation.md`](./nivel-2-rq2-ablation.md)):**
> con pre-entreno igualado y eval de 2 partidos, el snapshot v0 TAMBIÉN muestra ventaja
> de fine-tune — el contraste v0/v1 de abajo no replica. Lo transferible no es la
> profundidad temporal sino el **uso de las features de velocidad**. Este doc se
> conserva como registro del camino.

> Ejecutado 2026-07-24, ~$0. Reproducible: `scripts/nivel2_v1_temporal.py`. Misma
> batería que v0/few-shot para comparación directa: cada jugador pasa de un estado
> instantáneo (x,y,vx,vy) a una **trayectoria de 1 s** (5 pasos × x,y,vx,vy = 21 dims);
> misma arquitectura DeepSets y mismos presupuestos de optimización.

## Resultados (error mediano, fracciones de campo)

| | v0 por-frame | **v1 temporal** |
|---|---|---|
| in-domain fútbol (partido no visto) | 0.126 | **0.101** (corr +0.92/+0.94) |
| zero-shot baloncesto | 0.325 | 0.333 |
| in-domain basket full (~255 min, scratch) | 0.170 | 0.170 |

**Few-shot (finetune desde fútbol vs scratch, mediana de 3 semillas):**

| Presupuesto | v0: ft / scratch | v1: ft / scratch |
|---|---|---|
| 1 min | 0.266 / **0.241** | **0.260** / 0.270 |
| 5 min | 0.237 / **0.233** | **0.233** / 0.236 |
| 30 min | 0.227 / **0.218** | **0.201** / 0.219 |

En v0 el scratch ganaba siempre; **en v1 el finetune gana en todos los presupuestos**, y
en 30 min la mejora es consistente semilla a semilla (0.198<0.200, 0.209<0.229,
0.201<0.219).

## Lectura

1. **La estructura temporal es lo que transfiere.** Con representación por-frame, el
   pre-entreno futbolero no aportaba nada (incluso estorbaba); con trayectorias de 1 s,
   se vuelve un prior útil para aprender baloncesto con pocos datos. La regularidad
   deporte-general no está en las posiciones sino en **cómo evoluciona el movimiento
   coordinado** alrededor del objeto latente.
2. **Pero solo como prior de fine-tuning, nunca zero-shot** (0.333 ≈ v0 0.325, peor que
   el centroide sin entrenar 0.227). La dinámica transferible necesita calibrarse a la
   escala/ritmo del deporte destino.
3. El contexto temporal también mejora in-domain (~20%: 0.126 → 0.101), consistente con
   la literatura especialista (los SOTA son todos temporales).
4. Eco curioso del Nivel 1: también en los VLMs el "efecto temporal" no era leer la
   trayectoria (el orden no importaba) — aquí la dinámica sí aporta, pero como
   *representación aprendida*, no como lectura explícita del movimiento.

## Control de warm-start (¿transfer genuino o solo un init entrenado?)

Explicación alternativa: el input de 21 dims es más difícil de optimizar desde cero con
pocas muestras, y *cualquier* init entrenado ayudaría. Control: pre-entrenar la misma
arquitectura en fútbol con **targets permutados** (misma distribución de entrada,
mapping destruido) y repetir el few-shot (`scripts/nivel2_control_warmstart.py`):

| Presupuesto | fútbol | permutado | scratch |
|---|---|---|---|
| 1 min | **0.260** | 0.273 | 0.279 |
| 5 min | **0.233** | 0.235 | 0.236 |
| 30 min | **0.201** | 0.210 | 0.216 |

El init permutado queda entre medias: **~1/3 de la ventaja era warm-start genérico,
pero el pre-entreno real bate al control en los tres presupuestos** → existe componente
deporte-general genuino (~2/3 de la ventaja), aunque modesto.

## Consolidación (4 partidos fuente, 2 partidos eval, 5 semillas)

Re-ejecución a mayor escala (`scripts/nivel2_consolidate.py`: fuente = 2 Metrica + 2
SkillCorner, 75.8k muestras; eval = 2 partidos SportVU, 80.6k; pool few-shot = 4
partidos; 5 semillas):

| Presupuesto | fútbol | permutado | scratch |
|---|---|---|---|
| 1 min | **0.282** | 0.288 | 0.288 |
| 5 min | 0.246 | **0.240** | 0.247 |
| 30 min | **0.218** | 0.228 | 0.234 |

(zero-shot 0.347 · full pool 4 partidos 0.158)

**Lectura consolidada (la definitiva):** la ventaja del init futbolero **aguanta a
30 min** (4/5 semillas baten tanto al permutado como al scratch), pero a presupuestos
bajos las tres condiciones quedan dentro del ruido (a 5 min "gana" el permutado). El
transfer genuino de dinámica **existe pero es pequeño (~4-7%) y solo emerge con
suficiente calibración en el deporte destino**. Ni "no transfiere nada" (v0) ni "la
dinámica transfiere" a secas: transfiere un poco, tarde, y sobrevive al control.

## ¿Contradice esto el "nadie lee el movimiento" del Nivel 1? No — son dos cosas

- **Nivel 1** (VLMs congelados, inferencia, píxeles): ¿puede un VLM *leer* el movimiento
  de una secuencia de imágenes? No — el orden era irrelevante; solo usan vistas extra.
- **Nivel 2** (especialista, entrenamiento, coordenadas de campo): ¿la señal dinámica
  existe y lo *aprendido* de ella se reutiliza entre deportes? Sí (con el control de
  arriba).

Encajan: **el balón está escrito en la dinámica colectiva (N2), pero los VLMs actuales
no saben extraerla de píxeles (N1)** — el v1 identifica exactamente la señal que los
VLMs desaprovechan.

## Caveats

- Magnitudes modestas (~4-8% sobre scratch) con 3 semillas; dirección consistente en
  los tres presupuestos y semilla a semilla en 30 min, pero para el paper hay que
  consolidar con más partidos (SkillCorner + más SportVU) y más semillas.
- Un solo partido de entrenamiento de fútbol; el prior podría crecer con más datos de
  origen.

## Arco del Nivel 2 hasta aquí (Fase 0 → few-shot → v1)

1. En broadcast, la geometría posicional es anti-informativa (cámara).
2. En campo, la geometría funciona y lo aprendido la mejora — pero no cruza deportes.
3. **Lo único aprendido que cruza deportes es la dinámica temporal, y solo como prior
   de fine-tuning.** El prior trivial (centroide) sigue siendo la vara para presupuestos
   bajos.
