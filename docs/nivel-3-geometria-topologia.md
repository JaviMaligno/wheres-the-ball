# Nivel 3 — Geometría, topología y límites de información

> Estado: **ideas** (el más especulativo; se concreta con los resultados de los
> niveles 1 y 2). Contexto general: ver [README](./README.md).

## 1. Pregunta de investigación

**¿Es la posición del balón una función de la geometría de la configuración de
jugadores — y hasta qué punto esa función es interpretable y tiene límites de
información caracterizables?**

Dos vertientes complementarias:

- **Vertiente geométrica (explicativa):** ¿cuánto del rendimiento de un modelo deep
  (Nivel 2) se recupera con un conjunto pequeño de features geométricos/topológicos
  interpretables? Si ~10 features recuperan el 80%, el resultado es "la inferencia
  del balón *es* geometría", con valor tanto teórico como práctico (modelos baratos
  e interpretables).
- **Vertiente informacional (límites):** caracterizar la posterior
  p(balón | jugadores): cuándo es unimodal y estrecha (jugada trenzada), cuándo
  multimodal (pase largo en vuelo). Cota inferior del error alcanzable por
  *cualquier* modelo, y calibración frente a ella.

## 2. Caja de herramientas matemática (candidatas)

### Geometría computacional

- **Voronoi/Delaunay** sobre posiciones de jugadores: hipótesis de que el balón vive
  mayoritariamente en la célula del poseedor o en fronteras entre dominios de
  equipos rivales.
- **Pitch control** (Spearman): campo escalar de probabilidad de control; el balón
  como atractor de la dinámica del campo. Feature natural: posición del balón
  relativa al gradiente de pitch control.
- **Campos vectoriales:** velocidades (y orientaciones corporales, si el dataset las
  tiene) como campo vectorial muestreado en las posiciones de los jugadores. El
  balón como "punto de convergencia": mínimo de divergencia del campo interpolado, o
  punto que minimiza la suma de distancias a las semirrectas de orientación. Hay
  aquí una formulación elegante tipo índice de campo vectorial (los jugadores
  "apuntan" al balón ⇒ el campo tiene un punto singular de índice positivo cerca de
  él) que puede dar el teorema decorativo-pero-honesto del paper.

### Topología

- **Homología persistente** de la nube de jugadores (filtración de Vietoris–Rips,
  por equipo y conjunta): ¿los ciclos H1 (anillos de jugadores) encierran al balón
  en disputas y balones divididos? ¿La aparición/muerte de ciclos precede a eventos
  (pase que rompe el anillo)?
- **Persistencia de subniveles del pitch control** como descriptor del "paisaje" del
  campo, más estable que los diagramas crudos.
- Herramientas: GUDHI, giotto-tda; mplsoccer para pitch control y visualización.
- **Advertencia honesta:** con 10–22 puntos por frame, los diagramas de persistencia
  son pequeños y ruidosos. La persistencia luce más como descriptor de *secuencias*
  (evolución temporal de la formación) que de frames sueltos. No forzar TDA donde
  una distancia euclídea baste — el revisor lo huele.

### Información e incertidumbre

- Estimar p(balón | configuración) con un modelo generativo condicional (p. ej.
  normalizing flow o mixture density network sobre el output del modelo del
  Nivel 2) y medir su entropía por estrato de estado del balón.
- **Cota empírica de inferibilidad:** error del mejor modelo alcanzable frente a
  entropía de la posterior; mapa del campo de "dónde es inferible el balón".
- Conexión con la referencia humana del Nivel 1: ¿la incertidumbre humana y la del
  modelo crecen en los mismos ítems?

## 3. Diseño experimental (boceto)

1. **Pipeline de features:** para cada frame de los datasets del Nivel 2, calcular
   el vector de features geométricos/topológicos (Voronoi, pitch control, campo de
   velocidades, resúmenes de persistencia).
2. **Modelos interpretables:** regresión (gradient boosting / lineal) de la posición
   del balón sobre esos features. Comparar contra el deep del Nivel 2 con las
   métricas comunes.
3. **Análisis de atribución cruzado:** ¿qué miran las atenciones del transformer del
   Nivel 2? ¿Se alinean con los features geométricos (p. ej. atiende a los jugadores
   de la frontera de Voronoi)? Esto une las dos vertientes.
4. **Posterior y multimodalidad:** entrenar el modelo generativo condicional,
   evaluar NLL y calibración por estrato; identificar los regímenes de
   inferibilidad.
5. **(Si hay resultado) formalización:** enunciar el resultado del campo vectorial /
   convergencia con condiciones realistas y demostrarlo en el caso idealizado.

## 4. Narrativa y encaje

- Si la vertiente geométrica funciona: *"Ball inference is (mostly) geometry"* —
  paper metodológico con mensaje interpretable, target tipo revista de sports
  analytics o venue ML aplicado.
- Si domina la vertiente informacional: *"The limits of social inference of latent
  objects"* — framing general que conecta con percepción amodal y conducción
  autónoma (inferir agentes/objetos ocluidos por el comportamiento de terceros), y
  con la literatura de theory-of-mind computacional.
- El puente con conducción es narrativo salvo que aparezca colaboración con datos
  reales de ese dominio; no prometerlo como contribución experimental.

## 5. Riesgos y limitaciones

- **TDA cosmética:** el mayor riesgo del nivel. Regla autoimpuesta: cada feature
  topológico debe batir a su contraparte geométrica simple (distancias, densidades)
  en ablación, o se cae del paper.
- **Circularidad con pitch control:** los modelos de pitch control a veces usan la
  posición del balón como input. Usar variantes que no la usen, o documentar la
  dependencia.
- **La orientación corporal** (el feature teóricamente más bonito: "todos miran al
  balón") no está en la mayoría de datasets de tracking; puede requerir estimación
  de pose desde vídeo, con su ruido.
- **Alcance teórico honesto:** lo demostrable formalmente será sobre un modelo
  idealizado (jugadores como agentes que orientan su movimiento hacia el objetivo).
  El valor está en que el caso idealizado explique los datos, no en el teorema solo.

## 6. Dependencias

- Requiere los datasets normalizados y el modelo especialista del Nivel 2 (como
  techo y como objeto de análisis de atribución).
- Requiere los estratos y (si se hizo) la referencia humana del Nivel 1 para el
  análisis de incertidumbre comparado.
