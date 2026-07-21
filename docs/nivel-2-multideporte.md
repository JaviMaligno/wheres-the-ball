# Nivel 2 — Estudio multi-deporte: ¿qué transfiere al cambiar de deporte?

> Estado: **boceto** (se concreta tras los resultados del Nivel 1).
> Contexto general: ver [README](./README.md). Diseño previo: [Nivel 1](./nivel-1-benchmark-vlm.md).

## 1. Pregunta de investigación

**¿Cuánto de la habilidad de inferir el objeto de juego es específica del deporte y
cuánto es estructura colectiva universal?** Traducción experimental de la pregunta
original "¿hasta qué punto necesitas conocer las reglas del deporte?".

Sub-preguntas:

- **RQ1.** Un modelo especialista entrenado en fútbol, ¿infiere el balón en
  baloncesto/balonmano zero-shot? ¿Y con fine-tuning ligero (few-shot)?
- **RQ2.** ¿Qué features llevan la señal transferible? Ablación: solo posiciones /
  +velocidades / +aceleraciones / +orientación corporal / +identidad de rol
  (portero, pívot…).
- **RQ3.** ¿Hay deportes "más inferibles" que otros? (hipótesis: cuanto más continuo
  el juego y más acoplados los jugadores al balón — balonmano, baloncesto — más
  fácil; rugby con el balón escondido en el ruck, caso límite interesante).
- **RQL (enlace con Nivel 1).** ¿Los VLMs generalistas muestran el mismo patrón de
  transferencia que los especialistas, o transfieren mejor porque su conocimiento es
  semántico y no geométrico?

## 2. Hipótesis

- **H1.** La transferencia zero-shot entre deportes de invasión es sustancial
  (>50% del rendimiento in-domain) porque la señal dominante es geométrica:
  convergencia, orientación y densidad alrededor del objeto latente.
- **H2.** Las velocidades importan más que las posiciones; la orientación corporal
  es el feature más transferible (mirar al balón es universal).
- **H3.** El few-shot (minutos de datos del deporte destino) recupera casi todo el
  gap — "conocer las reglas" equivale a calibrar la geometría, no a reaprenderla.

## 3. Datos candidatos

| Deporte | Fuente | Estado |
|---|---|---|
| Fútbol | SoccerNet GSR; Metrica Sports open data; SkillCorner open data | Abierto, suficiente |
| Baloncesto | NBA SportVU (temporadas filtradas públicas) | Abierto con reservas de licencia — verificar |
| Balonmano | Sin tracking abierto conocido | Probablemente extraer de vídeo (YOLO+ByteTrack) — ruido añadido |
| Rugby | Sin tracking abierto conocido | Igual que balonmano; opcional |

Decisión pragmática: **fútbol + baloncesto** como núcleo del paper; balonmano/rugby
como estudio de caso cualitativo si la extracción de tracking resulta viable.

Normalización entre deportes: coordenadas de campo normalizadas por dimensiones,
velocidades en unidades campo/segundo, sin identidad de equipo salvo ataque/defensa.

## 4. Modelos

- **Especialista propio ligero:** Set Transformer o GNN sobre el grafo de jugadores
  (equivariante a permutaciones, agnóstico al número de jugadores — imprescindible
  para cruzar deportes: 22 vs 10 vs 14 jugadores). Arquitectura inspirada en Kim et
  al. 2023 pero minimalista y reproducible.
- **Referencias:** TranSPORTmer si es reproducible con nuestro presupuesto; los
  baselines geométricos B1–B4 del Nivel 1 (que son 100% transferibles por
  construcción y forman la vara de medir de la hipótesis geométrica).
- **VLMs del Nivel 1** sobre una submuestra, para RQL.

## 5. Diseño experimental

Matriz entrenamiento × evaluación (in-domain en diagonal, transferencia fuera):

|  | eval fútbol | eval basket | eval balonmano* |
|---|---|---|---|
| train fútbol | in-domain | zero-shot / few-shot | zero-shot |
| train basket | zero-shot / few-shot | in-domain | zero-shot |
| train ambos | — | — | zero-shot |

(*si hay datos)

- Few-shot: fine-tuning con {1, 5, 30} minutos de juego del deporte destino.
- Ablaciones de features (RQ2) sobre la diagonal y sobre la mejor celda de
  transferencia.
- Métricas: las del Nivel 1 (error mediano, acierto a umbral, poseedor,
  calibración) + **ratio de transferencia** = rendimiento zero-shot / in-domain.

## 6. Limitaciones y riesgos

- **Licencias**: SportVU y datos de proveedores tienen restricciones de uso y
  publicación; revisar antes de comprometerse a publicar el dataset.
- **Confusión extracción-vs-inferencia**: en deportes donde extraemos tracking de
  vídeo, el error de tracking contamina el resultado. Reportar calidad de tracking
  por deporte y, si es posible, evaluar también con tracking degradado
  artificialmente en fútbol para separar los efectos.
- **Frecuencias de muestreo distintas** entre datasets (25 Hz vs 10 Hz…):
  homogeneizar y documentar.
- **El resultado negativo también vale**: si no hay transferencia, la conclusión "la
  inferencia del balón es específica del deporte, las reglas importan más que la
  geometría" contradice H1 y alimenta igualmente el Nivel 3.

## 7. Entregables

1. Código + modelo ligero reproducible (repo del experimento).
2. Paper de workshop (candidatos: SoccerNet workshop, CVsports @ CVPR, MLSA @
   ECML-PKDD).
3. Features y matrices de transferencia que sirven de insumo directo al análisis
   geométrico del Nivel 3.

## 8. Dependencias del Nivel 1

- Si en el Nivel 1 el baseline geométrico B4 queda cerca de los VLMs, la hipótesis
  geométrica gana peso y este nivel debe centrarse en las ablaciones (RQ2).
- Los estratos difíciles del Nivel 1 (balón en vuelo) definen dónde mirar la
  transferencia: es plausible que la posesión transfiera y el vuelo no.
