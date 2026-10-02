# AGENTS.md — wheres-the-ball

Instrucciones canónicas para agentes de código (Claude Code, Codex, etc.). `CLAUDE.md`
solo importa este fichero.

## Propósito

Investigación: ¿puede un VLM generalista inferir dónde está el balón (oculto) a partir de
la configuración y el movimiento de los jugadores? Escalera de tres niveles documentada en
`docs/` (Nivel 1 benchmark VLM, Nivel 2 multideporte con especialistas ligeros, Nivel 3
geometría/topología/información). El resultado final es un paper en `paper/`.

Idioma: docs y README en español; código, docstrings y paper (`paper/*.tex`) en inglés.

## Estructura

- `src/wheres_the_ball/` — paquete (hatchling): `data/` (loaders: football-ball-detection,
  SoccerNet tracking, field tracking, ball state), `masking/` (inpainting del balón),
  `models/` (clientes VLM: `azure_gpt.py`, `anthropic_claude.py`, `base.py`),
  `prompts/`, `baselines/` (geométricos), `features/geometry.py`, `eval/` (métricas, viz).
- `scripts/` — un script por experimento, nombrado por fase/nivel (`fase0_*`, `fase1_*`,
  `nivel2_*`, `nivel3_*`, `nivel35_*`) y por análisis del paper (`paper_*`,
  `make_figures*.py`). Cada uno documenta su uso en el docstring de cabecera.
- `modal_app/` — jobs GPU en Modal (`qwen_vl.py`, `cnn_pixels.py`).
- `docs/` — diseño y resultados por fase/nivel (`fase-*`, `nivel-*`), revisiones del paper
  (`paper-review-synthesis.md`, `paper-relatedwork-proposal.md`).
- `paper/` — `main.tex` (una columna, revisión) y `main_twocol.tex` (dos columnas)
  comparten `abstract.tex` + `body.tex`; `refs.bib`; `paper/*.json` son los resultados
  numéricos citados; `paper/figures/` lo genera `scripts/make_figures_paper.py`.
- `data/` y `results/` — ignorados por git (grandes / regenerables). Los scripts leen y
  escriben con rutas relativas a la raíz (`results/fase1`, `results/nivel3`, ...).

## Setup y comandos

```bash
uv sync                          # crea .venv e instala dependencias (Python >=3.11)
cp .env.example .env             # credenciales: AZURE_API_BASE/KEY/VERSION, ANTHROPIC_API_KEY, HF_TOKEN
source ../CooperBench/azure_env.sh   # alternativa habitual para exportar las vars de Azure
```

Ejecutar siempre desde la raíz del repo (rutas relativas):

```bash
uv run python scripts/fase0_smoke.py --n 12        # smoke test Fase 0 -> results/fase0/
uv run python scripts/fase1_run.py [--limit N] [--overwrite] [--no-leak]
uv run python scripts/paper_scale_core.py          # análisis del paper (ver docstrings de paper_*.py)
uv run python scripts/make_figures_paper.py        # figuras -> paper/figures/
uv run modal run modal_app/qwen_vl.py [--prompt-variant informed]
uv run modal run modal_app/cnn_pixels.py --step mask|train
```

No hay suite de tests ni linter configurados. El paper se compila con LaTeX
(`main.tex` / `main_twocol.tex`); no hay Makefile.

## Convenciones y gotchas

- Los harness de predicción escriben incrementalmente (p. ej. `results/fase1/predictions.json`)
  y al relanzar saltan items ya predichos salvo `--overwrite`: no borrar resultados para
  "reintentar".
- `numpy<2` y `torch<2.3` están fijados a propósito (últimas wheels de torch para macOS Intel).
- Inpainting profundo (LaMa) no es dependencia del proyecto: instalarlo aparte
  (`uvx iopaint` o `simple-lama-inpainting`) para no romper el pin de pillow.
- En la Fase 0 las predicciones de Claude las producía el propio agente leyendo imágenes;
  el harness reejecutable (`scripts/fase0_claude_api.py`, `fase1_run.py`) sí necesita
  `ANTHROPIC_API_KEY`.
- `.gitignore` excluye `paper/*.aux|bbl|blg|log|out`, pero `paper/main.*` auxiliares ya
  están trackeados, así que aparecen modificados tras compilar.
- Las cifras del paper salen de los resultados escalados (especialistas: 12 partidos,
  leave-one-match-out; benchmark VLM: n=260 items), no de los números de 2 partidos de los
  primeros docs; actualizar `paper/*.json` y figuras desde
  los scripts `paper_*`, nunca a mano.
- Repo público (GitHub): no subir `.env`, credenciales ni datos de `data/`.
