# Nivel 1 — Diseño experimental: benchmark "Where's the ball?" para VLMs

> Estado: **diseño** (pendiente de ejecución).
> Contexto general y estado del arte: ver [README](./README.md).

## 1. Pregunta de investigación

**¿Puede un modelo de visión-lenguaje generalista localizar el balón que no ve,
infiriendo su posición a partir de la configuración y el movimiento de los
jugadores, como hace un espectador humano desde lejos?**

Sub-preguntas:

- **RQ1.** ¿Con qué precisión localizan los VLMs el balón oculto frente a baselines
  geométricos triviales y frente a un especialista supervisado (techo)?
- **RQ2.** ¿Mejora el rendimiento con contexto temporal (clip / secuencia de frames)
  frente a un frame aislado, igual que a un humano le ayuda ver la jugada moverse?
- **RQ3.** ¿Importa el conocimiento del juego? (mismo estímulo con/sin decirle qué
  deporte es, o con instrucciones tácticas explícitas tipo "los jugadores miran y
  corren hacia el balón").
- **RQ4.** ¿Saben los modelos *cuándo no saben*? (calibración de la incertidumbre
  declarada frente al error real).

## 2. Hipótesis

- **H1.** Los VLMs superan al azar y al centroide simple, pero quedan lejos del
  especialista supervisado.
- **H2.** El contexto temporal reduce el error de forma significativa (los humanos
  dependen del movimiento; esperamos lo mismo).
- **H3.** Las instrucciones con conocimiento del juego mejoran a los modelos débiles
  y apenas a los fuertes (que ya lo traen implícito).
- **H4.** El error es bimodal según el estado del balón: bajo en posesión/conducción,
  alto en pases largos y despejes (posterior multimodal).

Ambos desenlaces de H1 son publicables: si los VLMs funcionan, "la IA generalista
tiene intuición de espectador"; si fallan, "un humano cualquiera hace algo que los
mejores VLMs no pueden".

## 3. Datos

### Fuente principal

- **SoccerNet** (tracking + Game State Reconstruction): vídeo de broadcast con
  bounding boxes de jugadores y balón anotadas, y posiciones proyectadas a
  coordenadas de campo. Nos da: frames, ground truth del balón, y tracking de
  jugadores para los baselines geométricos.

### Construcción del conjunto de evaluación

- **Unidad de muestra:** un *ítem* = un frame objetivo (condición imagen) o un clip
  de 3–5 s terminando en el frame objetivo (condición vídeo/multi-frame).
- **Tamaño:** ~500 ítems para el artículo de blog (ver §7 para justificación
  estadística). Estratificados por **estado del balón**:
  - posesión/conducción (~40%)
  - pase corto en curso (~25%)
  - pase largo / despeje / balón en vuelo (~20%)
  - disputa / balón dividido (~15%)
- **Criterios de exclusión:** repeticiones, primeros planos, frames donde el balón
  está fuera de plano (esa condición se estudia aparte, no se mezcla), paradas de
  juego.

### Protocolo de enmascarado (crítico — riesgo de fuga)

Tres condiciones de ocultación, de más limpia a más realista:

1. **Oclusión natural** (preferida): seleccionar frames donde el balón ya no es
   visible o es ilegible (ocluido por jugadores, motion blur, tamaño < N px) pero la
   anotación de tracking existe (interpolada/anotada). Cero artefactos, máxima
   validez ecológica. Contra: la selección puede sesgar hacia jugadas congestionadas
   → controlar con la estratificación por estado del balón.
2. **Degradación global**: downscale + blur de todo el frame hasta que el balón sea
   ilegible pero los jugadores sigan siendo interpretables. Simula exactamente "ver
   el partido desde lejos". Sin artefactos locales que delaten la posición.
3. **Inpainting local**: borrar el balón con inpainting. Solo como condición
   secundaria, con **control de fuga**: pasar un detector (o el propio VLM) sobre
   los frames inpainted preguntando "¿ves el balón o algún artefacto de edición?";
   descartar ítems detectables. Reportar la tasa de descarte.

La condición primaria del artículo es la 1; la 2 sirve de réplica; la 3 solo si
hacen falta más datos en algún estrato.

### Formato de respuesta y ground truth

- El modelo responde con coordenadas normalizadas `(x, y)` en la imagen (0–1) +
  radio de incertidumbre + confianza declarada (0–100) + justificación breve.
- Ground truth: centro del bounding box anotado del balón.
- Para el análisis en metros: proyectar con la homografía de SoccerNet (cuando esté
  disponible) y reportar ambas escalas (píxeles normalizados y metros).

## 4. Sistemas a evaluar

### VLMs (API)

- Claude (último Sonnet/Opus disponible en el momento de ejecución)
- GPT (última versión multimodal)
- Gemini (última versión; interesante por soporte de vídeo nativo)
- Un VLM abierto (p. ej. Qwen-VL más reciente) como referencia open-source

Congelar versiones exactas de modelo y fecha en el registro de ejecución.

### Baselines geométricos (sin aprendizaje)

- **B0 — Azar** sobre el área de juego visible.
- **B1 — Centroide** de los jugadores detectados.
- **B2 — Centroide ponderado por velocidad** (los que corren más rápido pesan más;
  necesita 2 frames).
- **B3 — Jugador más rápido**: posición del jugador con mayor velocidad instantánea.
- **B4 — Voronoi + densidad**: centro de la célula de Voronoi del jugador en la zona
  de máxima densidad+velocidad. (Baseline "geométrico listo"; adelanta el Nivel 3.)

Los baselines usan el tracking de jugadores del dataset (no detección propia) para
que midan *información disponible*, no calidad de detección.

### Techo especialista

- Reproducir (o citar con números comparables) un modelo tipo Kim et al. 2023 /
  TranSPORTmer sobre el mismo conjunto, como cota superior de referencia. Si
  reproducirlo es caro, usar como techo el error publicado en condiciones análogas y
  señalarlo como limitación.

### Referencia humana (opcional pero muy recomendada)

- 5–10 personas "espectador normal" (conocen el fútbol, no expertos en análisis),
  mismo interfaz: ven el ítem, clican dónde creen que está el balón, dan confianza.
  ~100 ítems/persona (submuestra balanceada). Esto convierte el artículo de "los
  VLMs hacen X" a "los VLMs frente a la intuición humana", que es la narrativa
  original.

## 5. Prompts y condiciones experimentales

Matriz de condiciones (por modelo):

| Factor | Niveles |
|---|---|
| Contexto temporal | frame único / 4 frames (t−3s…t) / clip de vídeo (modelos que lo soporten) |
| Conocimiento del juego | prompt neutro ("localiza el objeto que los jugadores disputan") / prompt informado (deporte + reglas resumidas + heurísticas de atención: orientación, carreras, formación) |
| Enmascarado | oclusión natural / degradación global |

- Prompt base versionado en el repo del experimento; temperatura 0 o mínima;
  1 repetición por ítem y modelo (ampliar a 3 repeticiones en una submuestra para
  medir varianza intra-modelo).
- Respuesta en JSON estricto (usar tool/structured output donde exista).
- Coste estimado: ~500 ítems × ~6 condiciones × 4 modelos ≈ 12.000 llamadas con
  imágenes. Presupuestar y, si excede, recortar la matriz (p. ej. la condición
  vídeo solo en 2 modelos).

## 6. Métricas

- **Error de localización**: distancia euclídea al ground truth (px normalizados y
  metros). Reportar mediana e IQR (la distribución tendrá colas largas), no solo
  media.
- **Acierto a umbral** (estilo PCK): % de ítems con error < r para r ∈ {1 m, 3 m,
  5 m} (o equivalente en px cuando no haya homografía).
- **Acierto de poseedor**: ¿el jugador más cercano a la predicción es el poseedor
  real? (métrica robusta a error métrico moderado y alineada con cómo razona un
  humano).
- **Calibración**: correlación confianza declarada ↔ error; ECE adaptado a
  regresión; comparación del radio de incertidumbre declarado con el error real.
- **Desglose por estrato** de estado del balón (clave para H4).

## 7. Análisis estadístico

- Intervalos de confianza por **bootstrap** (ítems remuestreados, 10k réplicas).
- Comparaciones entre sistemas: test pareado por ítem (Wilcoxon sobre errores
  pareados) con corrección de Holm para comparaciones múltiples.
- Tamaño de muestra: con 500 ítems, un IC 95% sobre "acierto < 3 m" tiene semiancho
  ≈ 4 puntos porcentuales en el peor caso (p=0,5); suficiente para separar VLMs de
  baselines si la diferencia real es ≥ 10 puntos. Los estratos pequeños (~75 ítems)
  solo soportan conclusiones cualitativas — decirlo explícitamente en el artículo.
- Registrar TODO antes de ejecutar (este documento actúa de pre-registro informal):
  hipótesis, métricas, exclusiones. Cualquier análisis post-hoc se etiqueta como
  exploratorio.

## 8. Limitaciones conocidas del diseño

- **Un solo deporte** (fútbol): la generalización se pospone al Nivel 2 a propósito.
- **Broadcast ≠ vista de grada**: la cámara de TV sigue al balón, lo que en sí es
  una pista (el balón suele estar cerca del centro del encuadre). Mitigación:
  incluir B0' = "centro del frame" como baseline adicional; si los VLMs no lo
  superan claramente, el resultado está confundido y hay que reponderar ítems con
  balón descentrado.
- **Anotaciones interpoladas**: en oclusión natural el ground truth puede ser
  interpolado y tener error propio (~decenas de cm). Aceptable para umbrales ≥ 1 m.
- **Los VLMs cambian rápido**: fechar los resultados y publicar el benchmark de
  forma reejecutable importa más que el ranking puntual de modelos.
- **Posible memorización**: partidos de SoccerNet pueden estar en el entrenamiento
  de los VLMs. Mitigación parcial: la degradación global destruye detalles
  memorizables; comprobar con ítems de partidos recientes si es viable.

## 9. Entregables

1. **Repo del experimento** (separado de este sitio web): scripts de construcción
   del dataset, prompts versionados, harness de evaluación, análisis.
2. **Dataset de evaluación** (ítems + ground truth + estratos), publicable si las
   licencias de SoccerNet lo permiten (revisar; si no, publicar solo IDs + scripts).
3. **Artículo de blog** (EN/ES, con la skill `blog-writer`): narrativa desde la
   anécdota de la grada, resultados con visualizaciones (mapas de error sobre el
   campo, ejemplos cualitativos de aciertos/fallos con razonamiento del modelo).
4. **Decisión go/no-go para el Nivel 2**, con lo aprendido (¿qué estratos son
   difíciles? ¿la geometría trivial ya lo resuelve? ¿los VLMs razonan o adivinan?).

## 10. Plan de ejecución (fases)

1. **Fase 0 — viabilidad (1–2 días):** descargar una muestra de SoccerNet, verificar
   que existen suficientes frames de oclusión natural por estrato, probar el prompt
   con 10 ítems a mano en 2 modelos.
2. **Fase 1 — construcción del dataset (~1 semana):** selección, estratificación,
   enmascarado, control de fuga, congelar el conjunto.
3. **Fase 2 — harness + baselines (2–3 días):** implementar B0–B4, ejecutar sobre el
   conjunto congelado.
4. **Fase 3 — evaluación VLM (2–3 días):** ejecutar la matriz, con caché de
   respuestas y registro de versiones/costes.
5. **Fase 4 — (opcional) estudio humano (~1 semana en paralelo):** interfaz web
   simple (puede vivir en este mismo sitio), reclutar 5–10 personas.
6. **Fase 5 — análisis y artículo (~1 semana).**
