# Nivel 2 · H3 few-shot — ¿el pre-entreno en fútbol acelera aprender baloncesto?

> Ejecutado 2026-07-24, ~$0 (CPU local). Reproducible: `scripts/nivel2_fewshot.py`.
> Eval: el mismo partido SportVU del zero-shot (comparable con 0.325); few-shot desde
> **otros** dos partidos (sin fuga). 3 semillas por condición (trozos contiguos
> distintos), pasos de optimización igualados (600) entre presupuestos.

## Diseño

En cada presupuesto {1, 5, 30} minutos de metraje de baloncesto, dos modelos con
exactamente las mismas muestras:

- **finetune**: inicializado desde el especialista de fútbol (`specialist_v0.pt`).
- **scratch**: inicialización aleatoria.

Si finetune gana en presupuestos bajos → la geometría futbolera transfiere como prior.
Si empatan (o scratch gana) → el pre-entreno no aporta nada entre deportes.

## Resultados (error mediano en fracciones de campo; mediana de 3 semillas)

| Presupuesto | finetune (desde fútbol) | scratch |
|---|---|---|
| zero-shot (0 min) | 0.325 | — |
| 1 min | 0.266 | **0.241** |
| 5 min | 0.237 | **0.233** |
| 30 min | 0.227 | **0.218** |
| full (~255 min) | — | **0.170** |

Referencias: centro de cancha 0.369 · centroide sin entrenar 0.227 · vel-centroide 0.216
· in-domain fútbol del mismo modelo 0.126.

## Lectura

1. **El pre-entreno en fútbol no aporta nada** — scratch ≥ finetune en todos los
   presupuestos (leve transfer negativo si acaso). "Conocer el fútbol" no es un prior
   útil para localizar el balón en baloncesto, ni siquiera como inicialización.
2. **Un minuto de baloncesto vale más que 90 de fútbol**: 1 min desde cero (0.241) bate
   de largo el zero-shot del modelo futbolero completo (0.325).
3. **La vara del centroide es dura**: hasta ~30 min los modelos aprendidos apenas
   igualan al centroide sin entrenar (0.227); con el pool completo (~255 min) sí lo
   superan con claridad (0.170).
4. Con la Fase 0, la conclusión provisional del Nivel 2: **lo único que cruza deportes
   es el prior geométrico trivial (el balón vive cerca de la masa); todo lo aprendido
   encima es específico del deporte y no transfiere ni como init.** H1 (transferencia
   sustancial) y H3 (few-shot recupera el gap *gracias al pre-entreno*) quedan
   refutadas para este modelo.

## Caveats

- Modelo diminuto y por-frame: un modelo temporal mayor podría transferir
  representaciones de otro modo (es la prueba que definiría el paper). La ausencia de
  transfer aquí es informativa pero no cierra la puerta a arquitecturas más ricas.
- Varianza entre semillas visible en 1 min (scratch 0.24-0.31); la mediana de 3 semillas
  es estable en la dirección scratch ≥ finetune en todos los presupuestos.
- In-domain basket (0.170) peor que in-domain fútbol (0.126): el eje corto de la cancha
  está menos determinado (corr_y ~0.4) y SportVU incluye balón parado sin filtrar.

## Siguiente

- v1 temporal (ventana de frames) y re-testar el transfer con representación más rica.
- Ablación de features (RQ2): ¿posiciones vs velocidades? (la orientación corporal no
  está en el tracking — queda para datasets con pose).
- Matriz completa train×eval con más partidos (SkillCorner + más SportVU).
