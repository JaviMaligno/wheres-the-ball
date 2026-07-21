# Fase 1 · RQ3 — ¿ayuda el conocimiento del juego en el prompt?

> Misma muestra de-sesgada (n=92), single frame. Condición **neutral** (no nombra el
> deporte) vs **informado** (nombra fútbol + heurísticas de espectador: los jugadores
> orientan cuerpo/cabeza al balón, convergen hacia él, el poseedor va ligeramente
> adelantado, los defensas se sitúan entre balón y portería). Reproducible:
> `fase1_run.py --prompt informed --gpt-key gpt_informed --claude-key claude_informed`
> (y análogo para opus). Estimaciones puntuales (IC pendientes; ver auditoría).

## Resultado (neutral → informado)

| Modelo | corr_x | corr_y | far mediana | far win-rate |
|---|---|---|---|---|
| GPT-5.4 | 0.26 → **0.43** | 0.17 → **0.36** | 0.335 → 0.279 | 22/40 → 25/40 |
| Claude Opus 4.8 | 0.37 → 0.28 | 0.34 → 0.35 | 0.346 → 0.319 | 21/40 → 24/40 |
| Claude Sonnet 4.6 | −0.01 → 0.00 | 0.09 → 0.09 | 0.434 → 0.435 | 16/40 → 14/40 |

## Lectura (coherente con H3)

1. **GPT-5.4: el prompt informado ayuda claramente.** La correlación con el GT sube en
   ambos ejes (x 0.26→0.43, y 0.17→0.36) y mejora algo en descentrados. Tenía capacidad
   y margen; darle el marco del juego lo desbloquea. Con prompt informado, GPT iguala a
   Opus-neutral → **el prompt pesa tanto como la gama del modelo**.
2. **Claude Sonnet 4.6: no le hace nada** (correlación sigue nula). Su cuello de botella
   **no es falta de conocimiento del juego, sino de razonamiento visual-espacial**: por
   mucho que le expliquemos las reglas, no localiza mejor el balón.
3. **Claude Opus 4.8: apenas cambia** (x incluso baja un poco, y/far ~igual). Ya venía
   fuerte en neutral, así que el conocimiento explícito le aporta poco margen.

**Síntesis de la brecha (RQ3):** el prompt informado no reordena a los modelos de forma
uniforme — ayuda a quien tiene capacidad y margen (GPT), es neutro para el ya-fuerte
(Opus) e inútil para el que no razona la escena (Sonnet).

## Caveats

- Estimaciones puntuales a n=92 (far n=40); faltan IC bootstrap y test pareado
  informado-vs-neutral. El salto de GPT es grande y consistente (x, y, far) → creíble;
  los movimientos pequeños de Opus/Sonnet podrían ser ruido.
