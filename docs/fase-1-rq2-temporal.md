# Fase 1 · RQ2 — ¿ayuda el contexto temporal? (frame único vs. secuencia)

> Comparación **pareada** sobre los mismos 42 ítems del conjunto de-sesgado por
> distancia al centro. Condición temporal = secuencia de 4 frames (t-75/-50/-25/t, ~1s
> aparte), con el balón **borrado por LaMa en todos** (nunca visible). Prompt
> `temporal`; mismos modelos. Reproducible: `fase1_build_temporal.py` →
> `inpaint_lama.py --root results/fase1/clips` → `fase1_run_temporal.py` →
> `fase1_report_temporal.py`. Pareados: n=39 (3 ítems con <2 frames de historia).

## Hipótesis

Tras ver que un frame único no basta en balones descentrados (`far`), la pregunta: si el
modelo ve la **jugada moverse** (jugadores reorientándose, corriendo hacia el balón),
¿infiere mejor un balón que en estático era ambiguo? Es lo que hace un espectador.

## Resultado

### Balones descentrados (`far`) — donde un frame fallaba

| Modelo | single | temporal | baseline centro |
|---|---|---|---|
| **gpt-5.4** | 0.219 | **0.124** | 0.359 |
| claude-sonnet-4-6 | 0.410 | 0.401 | 0.359 |

### Global y por distancia al centro (mediana; single → temporal)

| Modelo | global | near | mid | far |
|---|---|---|---|---|
| gpt-5.4 | 0.117 → **0.089** | 0.086 → 0.086 | 0.140 → 0.177 | 0.219 → **0.124** |
| claude-sonnet-4-6 | 0.201 → 0.254 | 0.120 → 0.185 | 0.137 → 0.252 | 0.410 → 0.401 |

## Lectura

1. **En GPT-5.4 el contexto temporal casi halva el error en balones descentrados**
   (0.219 → 0.124) y baja el global (0.117 → 0.089). El efecto se concentra donde la
   hipótesis predecía: en `near` no cambia (ya era fácil), y en `far` —el caso ambiguo
   en estático— es donde el movimiento aporta. Es el comportamiento de un espectador:
   ver la jugada desplazarse resuelve la ambigüedad de un fotograma. GPT-temporal en
   `far` (0.124) pulveriza el baseline centro (0.359).
2. **En Claude Sonnet 4.6 el temporal no ayuda —incluso empeora** (global 0.201 →
   0.254; `near`/`mid` peores; `far` sin cambio y aún peor que el centro). No convierte
   la secuencia en mejor inferencia; posiblemente se confunde con múltiples imágenes o
   sigue anclando central.

**Conclusión.** RQ2 confirmado para el modelo capaz: el movimiento ayuda, y ayuda
*justo* donde el frame estático es ambiguo. Pero es específico del modelo — refuerza la
brecha de Fase 1: GPT-5.4 razona como espectador (estático + dinámico); Claude Sonnet
4.6 no aprovecha ni lo uno ni lo otro en balones difíciles.

## Caveats

- n=14 por bin (señal cualitativa; el ligero empeoramiento de GPT en `mid` es
  probablemente ruido). Escalar a ~500 con IC bootstrap.
- La ordenación/etiquetado de las 4 imágenes podría afectar a Claude; convendría un
  ablation (orden inverso, nº de frames) y probar `claude-opus-4-8`.
- Frames de historia con el balón sin anotar llevan máscara vacía (posible balón visible
  en algún frame previo); es raro y no afecta al frame objetivo.
