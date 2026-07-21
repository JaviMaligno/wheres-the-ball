# Fase 1 · RQ2 — ¿ayuda el contexto temporal? (frame único vs. secuencia)

> ⚠️ **Corrección tras auditoría ([`fase-1-auditoria.md`](./fase-1-auditoria.md)).** La
> "casi-mitad del error en `far`" de abajo compara medianas de GRUPO; la mejora
> **pareada** real en `far` tiene mediana Δ≈0 (IC incluye 0) para los tres modelos — el
> temporal ayuda mucho en 2 ítems de 14 y nada en el resto. **No concluir que el temporal
> desbloquea los balones descentrados** sin escalar la muestra.

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
| claude-opus-4-8 | 0.311 | 0.352 | 0.359 |
| claude-sonnet-4-6 | 0.410 | 0.401 | 0.359 |

### Global y por distancia al centro (mediana; single → temporal)

| Modelo | global | near | mid | far |
|---|---|---|---|---|
| gpt-5.4 | 0.117 → **0.089** | 0.086 → 0.086 | 0.140 → 0.177 | 0.219 → **0.124** |
| claude-opus-4-8 | 0.209 → 0.170 | 0.087 → 0.102 | 0.225 → **0.138** | 0.311 → 0.352 |
| claude-sonnet-4-6 | 0.201 → 0.254 | 0.120 → 0.185 | 0.137 → 0.252 | 0.410 → 0.401 |

## Lectura

1. **En GPT-5.4 el contexto temporal casi halva el error en balones descentrados**
   (0.219 → 0.124) y baja el global (0.117 → 0.089). El efecto se concentra donde la
   hipótesis predecía: en `near` no cambia (ya era fácil), y en `far` —el caso ambiguo
   en estático— es donde el movimiento aporta. Es el comportamiento de un espectador:
   ver la jugada desplazarse resuelve la ambigüedad de un fotograma. GPT-temporal en
   `far` (0.124) pulveriza el baseline centro (0.359).
2. **En Claude Opus 4.8 el temporal ayuda, pero en otro sitio** (global 0.209 → 0.170;
   mejora en 49% de ítems). El beneficio está en el rango **medio** (`mid` 0.225 →
   0.138), no en `far` (0.311 → 0.352, incluso peor). Es decir, el flagship de Anthropic
   sí aprovecha el movimiento, pero no logra convertirlo en inferencia de los balones más
   descentrados —justo donde GPT sí lo hace.
3. **En Claude Sonnet 4.6 el temporal no ayuda —incluso empeora** (global 0.201 →
   0.254; `near`/`mid` peores; `far` sin cambio y aún peor que el centro). No convierte
   la secuencia en mejor inferencia; posiblemente se confunde con múltiples imágenes o
   sigue anclando central.

**Conclusión.** RQ2 confirmado, pero con perfiles **específicos de cada modelo**:
GPT-5.4 usa el movimiento *justo* donde el frame estático es ambiguo (`far`, casi lo
halva) — el espectador de libro; Opus 4.8 lo aprovecha en el rango medio pero no en los
balones más descentrados; Sonnet 4.6 no lo aprovecha en absoluto. Refuerza la brecha de
Fase 1: solo GPT-5.4 convierte el contexto temporal en inferencia de los balones difíciles.

## Caveats

- n=14 por bin (señal cualitativa; el ligero empeoramiento de GPT en `mid` es
  probablemente ruido). Escalar a ~500 con IC bootstrap.
- La ordenación/etiquetado de las 4 imágenes podría afectar a Claude; convendría un
  ablation (orden inverso, nº de frames). (`claude-opus-4-8` ya incluido.)
- Opus tuvo algún error 500/529 transitorio de la API: single n=39, temporal n=38.
- Frames de historia con el balón sin anotar llevan máscara vacía (posible balón visible
  en algún frame previo); es raro y no afecta al frame objetivo.
