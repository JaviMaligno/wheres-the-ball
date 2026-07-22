# Fase 1 · RQ2 — ¿ayuda el contexto temporal? (n=92, autoritativo)

> Comparación **pareada** single-frame vs secuencia de 4 frames (t-75/-50/-25/t, ~1s
> aparte), con el balón borrado por LaMa en **todos** los frames. Muestra de-sesgada por
> distancia al centro, n=92 (87 con ≥2 frames de historia). Métrica primaria:
> **correlación pred↔GT** (bien alimentada); en `far`, mejora **pareada** con IC
> bootstrap. Reproducible: `fase1_build_temporal.py` → `inpaint_lama.py --root
> results/fase1/clips` → `fase1_run_temporal.py` (+ opus) → `audit_fase1.py`.
>
> ⚠️ Esto **reemplaza** un análisis previo a n=42 que estaba mal concluido (la "casi-mitad
> del error en far" era artefacto de medianas de grupo, movido por 2 ítems). Ver
> [`fase-1-auditoria.md`](./fase-1-auditoria.md).

## Resultado

**Correlación pred↔GT (single → temporal):**

| Modelo | corr_x | corr_y |
|---|---|---|
| GPT-5.4 | 0.26 → **0.47** | 0.17 → **0.31** |
| Claude Opus 4.8 | 0.37 → 0.34 | 0.34 → **0.20** |
| Claude Sonnet 4.6 | −0.01 → 0.12 | 0.09 → 0.08 |

**Mejora pareada en `far` (descentrados, n=38); Δ = error_single − error_temporal, >0 = temporal mejora:**

| Modelo | median Δ [IC95%] |
|---|---|
| GPT-5.4 | +0.022 [−0.003, +0.047] (borderline positivo) |
| Claude Sonnet 4.6 | −0.001 [−0.010, +0.047] (nulo) |
| Claude Opus 4.8 | −0.005 [−0.020, +0.002] (nulo/negativo) |

## Lectura

1. **GPT-5.4 sí aprovecha el contexto temporal.** La correlación con el GT sube en ambos
   ejes (x 0.26→0.47, y 0.17→0.31) y en balones descentrados mejora al borde de la
   significancia (Δ +0.022, IC casi excluye 0). El "efecto espectador" —ver la jugada
   moverse ayuda— **aguanta a escala** para GPT (a diferencia del falso positivo de n=42,
   ahora es la correlación bien alimentada la que lo sostiene).
2. **Opus 4.8 NO lo aprovecha, incluso empeora** (corr_y 0.34→0.20; far Δ negativo).
   Resultado llamativo: el mejor modelo en **imagen única** es el **peor usando la
   secuencia** — quizá su prior de un frame ya es bueno y las 4 imágenes le añaden ruido.
3. **Sonnet 4.6: apenas** (corr_x sube a 0.12 pero sigue débil; far nulo).

**Síntesis:** el beneficio del temporal es **específico de modelo y no monotónico con la
capacidad en imagen única**. GPT gana; Opus (el mejor en estático) pierde; Sonnet casi
nada. Es un hallazgo más interesante que un simple "el temporal ayuda".

## Caveats

- `far` n=38 (limitado por la cámara); el Δ de GPT es borderline. La señal fuerte y
  robusta es la correlación (n≈87).

## Mecanismo (ablation posterior)

El ablation ordenado-vs-desordenado-vs-formato
([`fase-1-ablation-temporal.md`](./fase-1-ablation-temporal.md)) mostró que **ningún
modelo usa el movimiento** (el orden de los frames es exactamente irrelevante): el
"efecto temporal" es un efecto **multi-vista** — a GPT le ayudan las vistas extra en
cualquier orden, a Opus le diluyen la lectura del frame objetivo.
