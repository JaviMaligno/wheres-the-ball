# Where's the ball?

**¿Puede un modelo de visión-lenguaje generalista localizar el balón que *no ve*,
infiriendo su posición a partir de la configuración y el movimiento de los jugadores,
como hace un espectador desde la grada?**

Viendo un partido desde lejos, sin distinguir bien el balón, una persona intuye dónde
está por cómo se mueven los jugadores. Este repo investiga si la IA generalista puede
hacer lo mismo — y hasta qué punto eso es geometría, conocimiento del juego, o ninguna
de las dos.

La pregunta binaria ("¿puede la IA inferir el balón?") ya está resuelta con modelos
**especialistas** (Maksai et al., Kim et al. 2023, TranSPORTmer). Lo que aquí se
explora está casi virgen: **VLMs generalistas**, generalización entre deportes, y la
geometría/topología subyacente. Ver [`docs/00-vision-general.md`](docs/00-vision-general.md).

## Estructura del proyecto

Escalera de tres niveles (cada uno alimenta al siguiente; el Nivel 1 es publicable solo):

| Nivel | Qué | Documento |
|---|---|---|
| 1 | Benchmark "Where's the ball?" para VLMs | [docs/nivel-1-benchmark-vlm.md](docs/nivel-1-benchmark-vlm.md) |
| 2 | Estudio multi-deporte con especialistas ligeros | [docs/nivel-2-multideporte.md](docs/nivel-2-multideporte.md) |
| 3 | Geometría, topología y límites de información | [docs/nivel-3-geometria-topologia.md](docs/nivel-3-geometria-topologia.md) |

Estado actual: **Fase 0 — viabilidad** del Nivel 1. Ver
[`docs/fase-0-viabilidad.md`](docs/fase-0-viabilidad.md).

## Setup

```bash
uv sync                     # crea .venv e instala dependencias
cp .env.example .env        # rellenar credenciales (ver abajo)
```

Credenciales de modelos (el patrón habitual de estos experimentos):

```bash
source ../CooperBench/azure_env.sh   # exporta AZURE_API_BASE/KEY/VERSION (GPT vía Azure)
```

En la Fase 0, las predicciones de **Claude** las produce el agente Claude Code leyendo
las imágenes directamente (no hace falta `ANTHROPIC_API_KEY`). Para el harness completo
y reejecutable del artículo sí se necesita la key.

## Fase 0 — smoke test

```bash
uv run python scripts/fase0_smoke.py --n 12
```

Baja una muestra de [`martinjolif/football-ball-detection`](https://huggingface.co/datasets/martinjolif/football-ball-detection)
(HuggingFace, CC BY 4.0), elimina el balón por inpainting usando su bbox anotado como
máscara, ejecuta el/los VLM y compara el error de localización contra el baseline
"centro del frame". Salida en `results/fase0/`.

## Licencia

Código bajo MIT. Los datasets mantienen sus licencias de origen (ver docs).
