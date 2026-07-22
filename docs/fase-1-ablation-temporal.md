# Fase 1 · Ablation del efecto temporal — ¿movimiento, vistas o formato?

> Tras RQ2 a escala (el "temporal" ayuda a GPT y estorba a Opus), dos controles sobre
> las **mismas** secuencias enmascaradas (sin nuevo inpainting) separan el mecanismo:
>
> - **shuffled**: los 3 frames de historia en orden aleatorio (determinista por ítem),
>   objetivo siempre al final → destruye la coherencia de movimiento, conserva las vistas.
> - **lastonly**: solo el frame objetivo, pero con el prompt/formato de secuencia →
>   aísla el efecto del formato multi-imagen.
>
> Reproducible: `fase1_run_temporal.py --variant shuffled|lastonly` (GPT + Opus;
> Sonnet excluido por ser nulo). n=87 (secuencias) / 92 (lastonly).

## Resultados

**Correlación pred↔GT por condición:**

| Condición | GPT corr_x | GPT corr_y | Opus corr_x | Opus corr_y |
|---|---|---|---|---|
| single | +0.26 | +0.17 | +0.37 | +0.34 |
| temporal (ordenado) | +0.47 | +0.31 | +0.34 | +0.20 |
| **shuffled** (sin orden) | **+0.42** | **+0.41** | +0.37 | **+0.13** |
| **lastonly** (solo formato) | +0.38 | +0.14 | +0.28 | **+0.33** |

**Contrastes pareados (mediana Δ error; * = IC 95% excluye 0):**

| Contraste | GPT | Opus |
|---|---|---|
| temporal vs shuffled (¿importa el orden?) | +0.002 [−.004,+.007] | +0.000 [−.001,+.000] |
| single → lastonly (¿solo formato?) | −0.010 [−.019,−.001]* | −0.001 [−.007,+.000] |

## Lectura — el mecanismo

1. **Nadie lee el movimiento.** El contraste ordenado-vs-desordenado es el nulo más
   limpio del experimento (Δ≈0.000-0.002 en ambos modelos; las correlaciones de shuffled
   igualan o superan a las del temporal ordenado). La trayectoria no aporta nada: los
   modelos no integran la dinámica de la jugada.
2. **El "efecto temporal" de GPT es en realidad un efecto MULTI-VISTA.** Su ganancia
   sobrevive intacta al shuffle (y 0.17→0.41 desordenado): ver la escena varias veces
   —en cualquier orden— le permite triangular mejor. Con un solo frame en formato
   secuencia (lastonly) la corr_y no mejora (0.14≈0.17) → hacen falta las vistas, no el
   prompt. El beneficio es además heterogéneo: concentrado en una minoría de ítems
   difíciles (por eso las medianas pareadas apenas se mueven mientras la correlación
   sube).
3. **A Opus le estorban las imágenes extra, no el formato ni el orden.** Su corr_y cae
   igual con historia ordenada (0.20) que desordenada (0.13), pero con lastonly
   (formato secuencia, una imagen) se mantiene (0.33≈0.34). Las vistas adicionales
   *diluyen* su lectura del frame objetivo — lo contrario exacto que GPT.

**Titular honesto:** el contexto temporal como *movimiento* no lo usa nadie; lo que
distingue a los modelos es cómo agregan **múltiples vistas** de la misma escena — GPT
suma, Opus se diluye. (Y la intuición estática más fuerte sigue siendo la de Opus.)

## Caveats

- Las correlaciones por condición tienen IC ~±0.2 y se solapan entre condiciones; lo
  nítido son los contrastes pareados (orden ≈ 0 exacto) y el patrón consistente
  x/y × modelo. Confirmación fina requeriría ~n=180 (split train).
- lastonly-GPT empeora la mediana una pizca (−0.010*) → pequeño coste del formato
  secuencia en sí, irrelevante frente a los efectos multi-vista.
