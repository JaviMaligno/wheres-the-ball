# Fase 0 — Viabilidad (Nivel 1)

> Estado: **en ejecución**. Objetivo: validar el pipeline y el prompt end-to-end con
> ~10-15 ítems y 2 modelos, y emitir un veredicto go/no-go para la Fase 1.
> Diseño completo del Nivel 1: [`nivel-1-benchmark-vlm.md`](./nivel-1-benchmark-vlm.md).

## Qué valida la Fase 0

Un smoke test, **no** el experimento. Comprueba que:

1. Podemos obtener frames de fútbol con GT de balón sin fricción.
2. Podemos ocultar el balón de forma visualmente coherente y sin fuga de información.
3. El prompt produce respuestas parseables (coordenada + incertidumbre + confianza).
4. El pipeline VLM (Azure GPT + Claude) funciona end-to-end y devuelve un error medible.
5. Hay señal: ¿los modelos superan el baseline trivial "centro del frame"?

## Decisiones de la sesión

### Datos

**`martinjolif/football-ball-detection`** (HuggingFace, CC BY 4.0). Imágenes broadcast
reales con bbox del balón, descarga sin fricción (`hf`/`datasets`). Se eligió tras
descartar alternativas (investigación en el registro de abajo).

Limitaciones asumidas para Fase 0 (se resuelven en el dataset formal):

- **Imágenes sueltas** → no se prueba contexto temporal (RQ2).
- **Solo GT de balón** (no jugadores) → el baseline geométrico se reduce a "centro del
  frame"; los baselines de centroide/Voronoi (B1-B4) llegan con el dataset formal.
- **Sin homografía** → error en píxeles normalizados (0-1), no en metros.

Dataset formal (Fase 1): **SoccerNet** (en cuanto haya cuenta) como primario; plan B de
acceso libre **ISSIA-CNR** o **SoccerTrack v1** (ambos con GT de balón *y* jugadores).

### Enmascarado: inpainting con control de fuga

Ni blur global (poco natural, degrada a los jugadores) ni parche local (delata la
posición). Se **elimina el balón por inpainting** para que el fondo quede coherente:

- Máscara = bbox anotado del balón, **dilatado** para llevarse sombra/halo.
- Inpainting: `cv2.inpaint` (Telea) primero — el balón es diminuto sobre fondo de baja
  frecuencia (césped/gradas), así que el relleno suele ser impecable e instantáneo.
  Fallback a **LaMa** (`simple-lama-inpainting`) si no es coherente.
- **Control de fuga obligatorio**: cada imagen editada se pasa por un VLM-detector
  ("¿ves el balón o algún artefacto de edición?"); los ítems detectables se marcan y se
  reportan como tasa de descarte.

### Modelos

- **GPT**: `gpt-5.4` multimodal vía Azure (litellm, credenciales de `azure_env.sh`).
- **Claude**: en Fase 0 lo produce el agente Claude Code leyendo cada imagen (no hay
  `ANTHROPIC_API_KEY`); anotado que el harness escalable la necesitará.

### Formato de respuesta (JSON estricto)

```json
{"x": 0.0-1.0, "y": 0.0-1.0, "uncertainty_radius": 0.0-1.0,
 "confidence": 0-100, "rationale": "…"}
```

`(x, y)` normalizados a [0,1] sobre el frame enmascarado. GT = centro del bbox del balón.

### Métricas Fase 0

- Error de localización = distancia euclídea normalizada al GT (mediana + IQR).
- Baseline de referencia: **centro del frame** (0.5, 0.5) — captura el sesgo de la
  cámara de broadcast. Un VLM que no lo supere no aporta señal.
- Visualización GT vs predicción sobre cada imagen.

## Resultados (12 ítems, seed=1, prompt v0-neutral)

Error = distancia euclídea normalizada al centro del balón (menor = mejor).

| Sistema | n | mediana | IQR (q1–q3) | media |
|---|---|---|---|---|
| center (baseline) | 12 | 0.311 | 0.237–0.353 | 0.297 |
| GPT (`gpt-5.4`, API) | 12 | **0.177** | 0.063–0.290 | 0.208 |
| Claude (`claude-sonnet-4-6`, API) | 12 | 0.257 | 0.159–0.376 | 0.266 |

Lectura:

- **Ambos VLMs generalistas baten el baseline "centro del frame"** (GPT 0.177, Claude
  0.257 vs 0.311). **H1 confirmada** en el smoke test: la IA generalista tiene "intuición
  de espectador". **GPT-5.4 va por delante de Claude Sonnet 4.6**, aunque cada uno gana
  en distintos ítems (Claude clava el 000: 0.040 vs 0.066; el 009: 0.139 vs 0.220).
- En varios ítems GPT clava el balón sobre el jugador en posesión leyendo la orientación
  del grupo (008: 0.004; 002/003/007: < 0.09).
- **Ambos fallan en pases largos** (ítem 004: el balón está en la banda opuesta, lejos
  del cluster). Confirma cualitativamente la bimodalidad de **H4**: la posterior
  p(balón|jugadores) es determinada en juego trenzado y ambigua en balón en vuelo.
- **El cluster no siempre marca el balón** (ítem 006: bloque defensivo denso, pero el
  balón estaba con un jugador suelto). Buen recordatorio contra el baseline de centroide.

### Nota metodológica (resuelta)

La primera pasada usó "Claude-vía-agente" (el propio Claude Code leyendo las imágenes) y
rindió al nivel del baseline (mediana 0.288, en `claude_preds_agent.json`). Con **Claude
por API** (Sonnet 4.6) sobre los mismos ítems, el error baja a 0.257 y bate el baseline
→ confirmado que el proxy-agente **no** era comparable (resolución reducida + coords a
ojo). El harness del artículo usa Claude por API (`scripts/fase0_claude_api.py`,
adaptador `models/anthropic_claude.py`; soporta key normal `sk-ant-api` y token de
suscripción `sk-ant-oat`).

**Caveat de comparación justa**: `gpt-5.4` es flagship y `claude-sonnet-4-6` es gama
media. Para flagship-vs-flagship en Fase 1, evaluar también `claude-opus-4-8`.

### Control de fuga

1/12 ítems marcado (000: el detector VLM vio un balón de repuesto cerca de la banda
lejana, no el balón en juego). El paso de control funciona y es imprescindible: además,
el inpainting Telea deja artefacto sobre líneas blancas (ver decisión de enmascarado),
así que en el dataset formal esos ítems irán por LaMa o se descartarán.

## Veredicto go/no-go

**GO para la Fase 1.** El pipeline funciona end-to-end (descarga → inpainting → control
de fuga → VLM → métricas → viz), el enmascarado por inpainting es visualmente coherente,
y hay señal cuantitativa clara (GPT bate el baseline) *y* hallazgos cualitativos ricos
(bimodalidad por estado del balón). Ajustes que entran en Fase 1:

1. ~~**Claude por API** en igualdad con GPT~~ ✅ hecho: `claude-sonnet-4-6` vía API bate
   el baseline (0.257). Falta añadir `claude-opus-4-8` para flagship-vs-flagship.
2. **Dataset formal** con GT de jugadores (SoccerNet/ISSIA) para habilitar baselines
   B1–B4 y análisis en metros, y con estratos por estado del balón.
3. **LaMa** para ítems con el balón sobre líneas/estructuras.
4. Añadir el baseline **"centro del frame"** como control permanente (sesgo de broadcast).

## Registro de decisiones

| Fecha | Decisión |
|---|---|
| 2026-07-21 | Repo separado `wheres-the-ball`, público en GitHub. Docs de diseño migrados de la rama `claude/sports-object-prediction-4oejhu`. |
| 2026-07-21 | Fase 0 con `football-ball-detection` (HF) por fricción cero; SoccerNet/ISSIA para el dataset formal. |
| 2026-07-21 | Enmascarado por inpainting (OpenCV→LaMa) con control de fuga, en vez de blur/parche. |
| 2026-07-21 | Modelos Fase 0: Azure `gpt-5.4` + Claude (vía agente). |
| 2026-07-21 | **GO a Fase 1.** GPT bate el baseline (mediana 0.177 vs 0.311). Decisión clave: Claude debe ir por API en Fase 1 (el sustituto vía agente no es comparable). |
| 2026-07-21 | Claude cableado por API (`sk-ant-api` de `llm-language-limits/.env`). `claude-sonnet-4-6` bate el baseline (0.257); GPT-5.4 sigue por delante. Ambos VLMs > baseline → H1 confirmada en smoke. |
| 2026-07-21 | SoccerNet-Tracking verificado como dataset primario de Fase 1 (sin NDA); ver `fase-1-datos-soccernet.md`. |
