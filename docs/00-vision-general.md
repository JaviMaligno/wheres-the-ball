# ¿Dónde está el balón? — Inferencia del objeto de juego a partir del movimiento de los jugadores

> Documento maestro del proyecto de investigación/experimentos. Los diseños detallados
> de cada nivel están en sus documentos específicos:
>
> - [Nivel 1 — Benchmark "Where's the ball?" para VLMs](./nivel-1-benchmark-vlm.md)
> - [Nivel 2 — Estudio multi-deporte con modelos especialistas ligeros](./nivel-2-multideporte.md)
> - [Nivel 3 — Geometría, topología y límites de información](./nivel-3-geometria-topologia.md)

## Origen de la idea

Viendo un partido de fútbol desde lejos, sin ver bien el balón, una persona puede
intuir dónde está a partir del movimiento de los jugadores — y esa intuición incluso
ayuda a *encontrarlo* visualmente. La pregunta: **¿puede la IA hacer lo mismo?** Y en
particular, ¿puede hacerlo un modelo *generalista* que "conoce el juego de manera
natural" (reglas, naturaleza competitiva, intención de marcar), como una persona
normal, no un sistema especialista de tracking?

Extensiones naturales:

- Otros deportes (baloncesto, balonmano, rugby…): ¿hay que especializarse por
  deporte o la habilidad transfiere? ¿Basta conocer las reglas o hace falta contexto
  de jugadas/intención?
- Conexión con otros dominios de percepción de objetos latentes: conducción autónoma
  (¿por dónde va a aparecer ese coche ocluido?), detección de movimiento, robótica.
  Framing general: **inferencia social de objetos latentes** ("social inference of
  latent objects").

## Estado del arte (resumen)

La pregunta binaria "¿puede la IA inferir el balón a partir de los jugadores?" **ya
está respondida afirmativamente con modelos especialistas**. Hay que citar esta
literatura como punto de partida, no ignorarla:

| Trabajo | Qué hace | Relevancia |
|---|---|---|
| Maksai et al., *What Players do with the Ball* ([arXiv:1511.06181](https://arxiv.org/abs/1511.06181)) | Infiere trayectoria del balón en fútbol/vóley/basket con tracking de jugadores + restricciones físicas | Trabajo fundacional (~2016) |
| Kim et al. 2023 ([arXiv:2306.08206](https://arxiv.org/abs/2306.08206)) | Set Transformer + Bi-LSTM jerárquico: primero predice poseedor, luego trayectoria del balón, **solo** desde trayectorias de jugadores | La formulación más cercana a nuestra intuición |
| TranSPORTmer (Capellera et al., [arXiv:2410.17785](https://arxiv.org/abs/2410.17785)) | Transformer multi-tarea unificado; ~25% de mejora en inferencia de balón | Estado del arte especialista |
| PathCRF ([arXiv:2602.12080](https://arxiv.org/abs/2602.12080)) | Detecta eventos de balón sin ver el balón, infiriendo la ruta de posesión | Eventos sin balón |
| Multi-Modal Soccer Scene Analysis ([arXiv:2512.19528](https://arxiv.org/abs/2512.19528)) | Pre-entrenamiento enmascarado; infiere poseedor, estado y trayectoria a la vez | Enfoque enmascarado multimodal |
| Benchmarks VLM deportivos: SPORTU ([arXiv:2410.08474](https://arxiv.org/abs/2410.08474)), SoccerLens ([arXiv:2605.09598](https://arxiv.org/abs/2605.09598)), inteligencia espacial en deportes ([arXiv:2603.09896](https://arxiv.org/abs/2603.09896)) | Evalúan comprensión deportiva de VLMs | Ninguno cubre el task "balón enmascarado" |
| TDA en deporte: scouting de fútbol ([Research Square](https://www.researchsquare.com/article/rs-7756175/v1)), hockey ([arXiv:1409.7635](https://arxiv.org/abs/1409.7635)) | Homología persistente para similitud de jugadores / patrones de equipo | TDA existe en deporte, pero **no** para inferir el balón |

Motivación compartida por toda la literatura (y la nuestra): el balón es el objeto
más difícil de trackear del deporte — diminuto, rápido, constantemente ocluido
([arXiv:2311.05237](https://arxiv.org/abs/2311.05237)).

## Huecos identificados (dónde está el artículo)

1. **VLMs generalistas.** Nadie ha evaluado si un modelo multimodal generalista
   (Claude, GPT, Gemini, Qwen-VL…) puede localizar el balón oculto usando solo el
   conocimiento natural del juego. Es la pregunta original y está casi virgen.
   → **Nivel 1**.
2. **Generalización entre deportes.** Casi toda la literatura es fútbol. "¿Cuánto hay
   que especializarse?" se traduce en entrenar en un deporte y evaluar zero-shot en
   otros. Hipótesis: lo que transfiere es la *geometría* de la estructura colectiva,
   no las reglas. → **Nivel 2**.
3. **Geometría y topología.** ¿La posición del balón es una función (aprendible, y
   hasta qué punto *interpretable*) de la geometría de la configuración de jugadores?
   Voronoi/Delaunay, pitch control, homología persistente, campos vectoriales de
   velocidades/orientaciones. Si pocos features geométricos interpretables recuperan
   la mayor parte del rendimiento del deep learning, eso es un resultado. → **Nivel 3**.
4. **Límites de información e incertidumbre.** La posterior p(balón | jugadores) es
   multimodal (pelotazo largo = ambiguo; jugada trenzada = determinada). Medir
   calibración y caracterizar *cuándo* los jugadores determinan el balón es una
   contribución en sí misma, y conecta con conducción autónoma. → **Nivel 3**.

## Hoja de ruta

Escalera de tres niveles; cada nivel alimenta al siguiente y el Nivel 1 es publicable
por sí solo (blog) aunque los demás no salgan.

| Nivel | Formato objetivo | Horizonte | Documento |
|---|---|---|---|
| 1. Benchmark VLM "Where's the ball?" | Artículo de blog (experimentos), potencialmente dataset/benchmark público | Semanas | [nivel-1-benchmark-vlm.md](./nivel-1-benchmark-vlm.md) |
| 2. Estudio multi-deporte especialista | Paper de workshop (SoccerNet, MLSA, CVsports) | Meses | [nivel-2-multideporte.md](./nivel-2-multideporte.md) |
| 3. Geometría/topología + incertidumbre | Paper de investigación | Largo | [nivel-3-geometria-topologia.md](./nivel-3-geometria-topologia.md) |

## Riesgos y limitaciones globales

- **La novedad no es "¿puede la IA?"** — eso ya está respondido con especialistas. La
  venta del trabajo son los huecos 1–4. Citar siempre a Maksai/Kim/Capellera como
  punto de partida.
- **Localización precisa en VLMs.** Los VLMs actuales son notoriamente malos dando
  coordenadas exactas en imágenes. Resultado posible del Nivel 1: "los VLMs fallan
  estrepitosamente donde un humano acierta". Eso también es un artículo (y de los que
  se comparten): hay que diseñar el experimento para que ambos desenlaces sean
  informativos.
- **Datos multi-deporte.** Tracking abierto de balonmano/rugby es escaso; puede tocar
  extraer tracking de vídeo (trabajo extra y ruido añadido). Pragmático: empezar por
  fútbol + baloncesto.
- **Fuga de información en el enmascarado.** Si el balón se oculta con inpainting
  imperfecto, el modelo puede localizar el *artefacto* en vez de inferir. El protocolo
  del Nivel 1 dedica una sección a esto.
- **Comparación con humanos.** El paralelismo humano ("yo lo intuyo desde la grada")
  es la narrativa; validarlo requiere un mini-estudio con personas, que es opcional
  pero refuerza mucho el artículo.

## Registro de decisiones

| Fecha | Decisión |
|---|---|
| 2026-07-19 | Estructura en 3 niveles; Nivel 1 primero, con diseño experimental completo |
