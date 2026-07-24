# Nivel 2 · v1 temporal — lo que transfiere es la dinámica (y solo como prior)

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
