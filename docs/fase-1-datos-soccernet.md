# Fase 1 — datos: SoccerNet-Tracking (verificado)

> Verificación de descarga y formato hecha el 2026-07-21 con `pip install SoccerNet`
> (v0.1.62). Confirma que SoccerNet-Tracking es el dataset primario para la Fase 1.

## Descarga (sin NDA)

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="data/SoccerNet")
d.downloadDataTask(task="tracking", split=["test"])   # también "train", "challenge"
```

- **No requiere password / NDA.** El NDA de SoccerNet es solo para los vídeos de
  broadcast crudos (LQ/HQ), no para los clips de tracking.
- Tamaño: `test.zip` ≈ **8.1 GB** (49 clips con los frames como imágenes). `train` será
  mayor. Está en `data/` (gitignored).

## Formato (MOT Challenge)

Cada clip `SNMOT-XXX/`:

- `img1/NNNNNN.jpg` — 750 frames, 25 fps, 1920×1080.
- `gt/gt.txt` — líneas `frame,id,x,y,w,h,conf,cls,vis`. `(x,y)` = esquina superior
  izquierda del bbox; `w,h` = tamaño. Una línea por objeto y frame.
- `gameinfo.ini` — mapea cada `trackletID_N` a su rol: `player;<team>`, `goalkeeper`,
  `referee`, `ball`. **El balón es el tracklet marcado `ball`** (p. ej. en SNMOT-116,
  `trackletID_20 = ball;1`).
- `seqinfo.ini` — metadatos (nombre, frameRate, seqLength, dimensiones).

## Por qué es el dataset primario de la Fase 1

Frente a `football-ball-detection` (Fase 0: imágenes sueltas, solo GT de balón):

| Capacidad | football-ball-detection | SoccerNet-Tracking |
|---|---|---|
| GT de balón por frame | sí | **sí** (bbox ~12×11 px) |
| GT de jugadores | no | **sí** (habilita baselines B1–B4) |
| Secuencia temporal | no | **sí** (750 frames/clip → RQ2) |
| Estratos por estado del balón | manual | **derivable** (velocidad del balón entre frames) |
| Homografía a metros | no | vía **SoccerNet-GSR** (dataset complementario) |

## Pasos de construcción del conjunto (Fase 1)

1. Parsear `gt.txt` + `gameinfo.ini` por clip → trayectoria del balón y de cada jugador.
2. Seleccionar frames objetivo estratificados por estado del balón (posesión / pase
   corto / pase largo / disputa), usando la velocidad del balón entre frames.
3. Enmascarar el balón por inpainting (bbox como máscara; Telea, con LaMa para balón
   sobre líneas) + control de fuga VLM.
4. Condición temporal (RQ2): pasar los N frames previos al frame objetivo.
5. Baselines B0–B4 con las posiciones de jugadores del propio GT.
