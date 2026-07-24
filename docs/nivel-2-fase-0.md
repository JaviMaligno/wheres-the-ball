# Nivel 2 · Fase 0 — viabilidad: geometría, especialista v0 y primer zero-shot

> Ejecutada 2026-07-23/24. Coste ~$0 (todo geometría/entrenos locales). Reproducible:
> `scripts/geo_baselines.py` (B2-B4 sobre el eval de Nivel 1) y
> `scripts/nivel2_specialist.py` (DeepSets sobre coordenadas de campo).

## 1. B2-B4 en espacio de imagen: la geometría simple es ANTI-informativa

Los baselines geométricos ricos que el Nivel 1 dejó pendientes (velocidades
compensadas de cámara — se resta el desplazamiento medio de los jugadores), sobre el
mismo eval de-sesgado n=92:

| Sistema | corr_x | corr_y | far win-rate |
|---|---|---|---|
| centroide (B1) | **−0.58** | −0.20 | 7/40 |
| B2 centroide×velocidad | −0.46 | −0.10 | 10/40 |
| B3 jugador más rápido | −0.19 | +0.03 | 10/40 |
| B4 Voronoi+densidad | −0.37 | −0.07 | 8/40 |
| (GPT-5.4 / Opus 4.8) | +0.26 / +0.37 | +0.17 / +0.34 | 22/40 / 21/40 |

**Todas las señales posicionales están anti-correlacionadas con el balón en broadcast.**
Explicación: cuando el balón está descentrado es porque va *por delante* del juego
(pase/despeje) y la masa de jugadores queda al lado contrario del encuadre. Implicación
fuerte para el Nivel 1: los VLMs frontera logran correlación *positiva*, así que leen
algo que ningún agregado posicional contiene (orientación corporal, postura, semántica).
Implicación para el Nivel 2: la prueba justa de la geometría es en coordenadas de campo.

## 2. Datos verificados (posiciones, sin imagen — el especialista no la necesita)

| Fuente | Deporte | Contenido | Licencia |
|---|---|---|---|
| Metrica sample-data | fútbol | 3 partidos, 25 Hz, coords 0-1, balón | abierta |
| SkillCorner opendata | fútbol | ~10 partidos, coords campo, balón | propia (no comercial) |
| NBA SportVU 2015-16 | baloncesto | ~600 partidos, 25 Hz, jugadores x,y + balón x,y,z (pies, 94×50) | **sin licencia explícita** — uso académico estándar; no redistribuimos datos, solo scripts |

También descargado el split **train** de SoccerNet-Tracking (8.9 GB) para el futuro
techo especialista en espacio de imagen sin fuga (entrenamiento fuera de los clips de eval).

## 3. Especialista v0 (DeepSets) — in-domain y primer zero-shot

Modelo mínimo invariante a permutaciones y al nº de jugadores (por-jugador MLP →
mean+max pool → cabeza; input por jugador: x, y, vx, vy, equipo±1; salida: balón x,y).
Entrenado en **Metrica partido 1** (17.6k muestras, 12 epochs, minutos de CPU).

**In-domain (Metrica partido 2, no visto; error mediano en fracciones de campo):**

| Sistema | mediana | corr (x, y) |
|---|---|---|
| centro del campo | 0.352 | — |
| centroide | 0.231 | (+0.83, +0.84) |
| centroide×velocidad | 0.204 | (+0.85, +0.88) |
| **especialista v0** | **0.126** | (+0.86, +0.94) |

- **En coordenadas de campo la geometría funciona** (centroide corr +0.83 vs −0.58 en
  imagen) → la anti-correlación del §1 era la cámara, no la geometría.
- El especialista aprendido reduce ~45% el error del mejor agregado simple.

**Zero-shot a baloncesto (SportVU, 1 partido, 41.9k muestras):**

| Sistema | mediana | corr (x, y) |
|---|---|---|
| centro de cancha | 0.369 | — |
| centroide | **0.227** | (+0.85, +0.41) |
| centroide×velocidad | **0.216** | (+0.86, +0.45) |
| especialista v0 (entrenado en fútbol) | 0.325 | (+0.74, +0.38) |

**El giro:** lo aprendido NO transfiere (el especialista cae a apenas-mejor-que-centro),
pero la geometría simple transfiere perfectamente — el centroide sin entrenar es mejor
en baloncesto que el modelo entrenado en fútbol. **Lo universal es el núcleo geométrico
("el balón vive cerca de la masa en movimiento"); la capa aprendida es específica del
deporte.** Responde H1 con matiz y convierte a H3 (¿cuánto few-shot recupera el gap?) en
la siguiente pregunta.

## Caveats

- v0 es por-frame y diminuto (sin contexto temporal): su in-domain (~13 m equivalentes)
  está lejos del SOTA especialista (Kim et al. ~2.5-5 m con modelos temporales). Para la
  matriz de transferencia del paper hará falta v1 temporal.
- 1 partido de entrenamiento y 1 de baloncesto; escalar partidos antes de conclusiones
  finas. SportVU incluye balón parado/tiros libres (sin filtrar).
- corr_y baja en baloncesto para todos (~0.4): el eje corto de la cancha está menos
  determinado por la masa — consistente entre sistemas.

## Siguiente (Fase 1 del Nivel 2)

1. **Few-shot** (H3): fine-tuning con {1, 5, 30} min de SportVU → ¿se recupera el gap?
2. Especialista **v1 temporal** (ventana de frames) + más partidos (SkillCorner + más SportVU).
3. Matriz completa train×eval (fútbol/basket) + ablación de features (RQ2).
4. Techo en espacio de imagen con el split train de SoccerNet (cierra el Nivel 1).
