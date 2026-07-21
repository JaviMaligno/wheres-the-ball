# Fase 1 — resultados (SoccerNet-Tracking + LaMa, dataset de-sesgado)

> ⚠️ **Leer junto a [`fase-1-auditoria.md`](./fase-1-auditoria.md).** Con n=14 por bin,
> los cortes por distancia al centro y el ranking fino GPT/Opus/Sonnet **no son
> estadísticamente concluyentes** (IC bootstrap muy amplios). Lo robusto es la
> correlación con el GT (Sonnet ~0; GPT y Opus con señal). Las tablas de abajo son
> estimaciones puntuales, pendientes de escalar la muestra.

> Harness de Nivel 1 sobre SoccerNet-Tracking (split test), enmascarado **LaMa**,
> control de fuga, prompt `v0-neutral`, **frame único**. El conjunto está **balanceado
> por distancia del balón al centro** (14 near / 14 mid / 14 far) para neutralizar el
> sesgo de la cámara de broadcast (que sigue al balón). Reproducible:
> `fase1_build_dataset.py --balance center` → `inpaint_lama.py` → `fase1_run.py` →
> `fase1_report.py`.

## Por qué de-sesgar (el confound que casi nos engaña)

Una primera corrida (estratificada por estado del balón) daba "GPT bate al baseline
centro". Pero un diagnóstico pareado lo desmontó: GPT ganaba al centro solo en 55% de
los ítems, y **en balones descentrados nadie batía al centro** — la ventaja venía de que
la cámara centra el balón, así que "predecir el centro" ya acertaba mucho. Sin controlar
esto, el titular es un artefacto de la cámara.

Solución: construir el conjunto **balanceado por distancia al centro**, de modo que
"centro" sea mal predictor por diseño (en el tercio `far`, el baseline centro tiene
error 0.359). Así, ganar al centro en `far` es evidencia real de inferencia.

## Resultado principal — error por distancia al centro

| Sistema | near (n=14) | mid (n=14) | **far / descentrados (n=14)** |
|---|---|---|---|
| center | 0.101 | 0.201 | 0.359 |
| centroid (B1) | 0.159 | 0.201 | 0.450 |
| **gpt-5.4** | 0.086 | 0.140 | **0.219** |
| claude-opus-4-8 | 0.087 | 0.225 | 0.311 |
| claude-sonnet-4-6 | 0.120 | 0.137 | 0.410 |

(claude-opus n=13 en cada bin: 3 ítems caídos por errores 500 transitorios de la API.)

**Win-rate pareado (¿bate al centro ítem a ítem?)**

| Sistema | global | solo `far` |
|---|---|---|
| gpt-5.4 | 23/42 (55%) | **9/14 (64%)** |
| claude-opus-4-8 | 19/39 (49%) | 6/13 (46%) |
| claude-sonnet-4-6 | 20/42 (48%) | 6/14 (43%) |
| centroid | 9/42 (21%) | 3/14 (21%) |

## Lectura (sin confound de cámara)

1. **GPT-5.4 tiene intuición real de espectador.** En balones descentrados —donde el
   centro es inútil (0.359)— GPT baja el error a **0.219** y gana **64%** de los ítems.
   No es sesgo de cámara: infiere de verdad la posición del balón desde los jugadores.
2. **Claude Sonnet 4.6 no lo consigue.** En `far` es **peor que el centro** (0.410 vs
   0.359) y gana solo 43%: ancla cerca del centro/cluster y falla cuando el balón está
   genuinamente lejos.
3. **Claude Opus 4.8 queda en medio (resuelve el "gama vs Anthropic").** Iguala a GPT en
   balones centrados (`near` 0.087) y tiene la mejor tasa de acierto exacto (PCK@.05 0.26,
   poseedor 0.38 ≈ GPT), pero perfil **bimodal**: clava muchos y falla feo en otros, así
   que su mediana global (0.209) no destaca. En `far` **mejora claramente a Sonnet**
   (0.311 vs 0.410) y bate al baseline centro (0.359), pero **no alcanza a GPT** (0.219).
   Conclusión: el flagship de Anthropic sí tiene intuición (mucho más que su gama media),
   pero **GPT-5.4 mantiene la ventaja en el caso más difícil** (balón descentrado).
4. **El centroide de jugadores (B1) es el peor**, sobre todo en `far` (0.450). La
   geometría naíf no sirve — el balón no está en el centroide (motiva la geometría fina
   del Nivel 3).

Global (mediana): center 0.201 · centroid 0.239 · **gpt 0.117** · claude 0.201 ·
claude-opus 0.209 (media 0.226). PCK@.10: gpt 0.48 vs center 0.17. Acierto de poseedor:
gpt 0.40 ≈ opus 0.38 ≫ sonnet 0.24 vs center 0.21.

### Por estado del balón (metadato)

| Sistema | possession (10) | short_pass (22) | long_pass (10) |
|---|---|---|---|
| gpt | 0.389 | 0.096 | 0.119 |
| claude | 0.318 | 0.201 | 0.157 |
| center | 0.251 | 0.199 | 0.158 |

`possession` es sorprendentemente el peor estrato para GPT: en este conjunto son en su
mayoría balones de construcción en banda (lejos del centro y ambiguos desde un frame).

## Arreglos aplicados

- **Sesgo de cámara**: conjunto balanceado por distancia al centro (arriba).
- **Fuga por motion blur**: la máscara ahora se dilata con la velocidad del balón
  (`extra_pad_px ∝ speed`); la tasa de fuga bajó de 6/42 → **4/42**.

## Hecho después / pendiente

- ✅ `claude-opus-4-8` (comparación flagship): ver arriba (Opus entre Sonnet y GPT).
- ✅ **Condición temporal (RQ2)**: ver [`fase-1-rq2-temporal.md`](./fase-1-rq2-temporal.md)
  — el temporal casi halva el error de GPT en balones descentrados; Opus mejora en el
  rango medio; Sonnet no.
- Pendiente: VLM abierto (Qwen-VL); escalar a ~500 ítems con IC por bootstrap; prompt
  informado (RQ3); ablation temporal (orden/nº de frames).
