# Nivel 2 · RQ2 — ablación de features: ¿qué lleva la señal (transferible)?

> Ejecutado 2026-07-24, ~$0. Reproducible: `scripts/nivel2_ablation_rq2.py`. Variantes
> cortadas de las mismas trayectorias de 21 dims; mismo modelo/presupuestos; eval basket
> a 2 partidos; transfer = fine-tune vs scratch con 30 min (3 semillas).

## Resultados (error mediano)

| Variante (input por jugador) | in-domain fútbol | zero-shot basket | 30min ft | 30min scratch | ¿ft gana? |
|---|---|---|---|---|---|
| `pos_traj` (x,y)×5 | 0.122 | **0.274** | 0.231 | 0.233 | no (0/3) |
| `vel_traj` (vx,vy)×5 | 0.295 | 0.453 | 0.322 | 0.335 | sí (3/3, pero pésimo) |
| `single` (x,y,vx,vy)×1 — v0 | 0.113 | 0.333 | **0.216** | 0.233 | **sí (3/3)** |
| `full_traj` (x,y,vx,vy)×5 — v1 | **0.103** | 0.339 | 0.218 | 0.231 | **sí (3/3)** |

## Lecturas

1. **H2 refutada.** Las velocidades *solas* son casi inútiles (in-domain 0.295, zero-shot
   0.453 — peor que el centro). La señal central vive en las **posiciones**.
2. **Pero la parte transferible del aprendizaje es el USO de las velocidades.** La
   ventaja del fine-tune aparece exactamente en las variantes con velocidades (single,
   full, vel — 3/3 semillas) y desaparece en posiciones-solo. Coherente con el zero-shot:
   añadir velocidades EMPEORA el zero-shot (0.274→0.339; las escalas de velocidad son
   específicas del deporte y hay que recalibrarlas), pero una vez calibradas con ~30 min,
   el conocimiento pre-entrenado de cómo explotarlas paga.
3. **La profundidad temporal en sí es marginal para el transfer**: el snapshot con
   velocidades (v0) gana tanto como la trayectoria completa (0.216 vs 0.218); la
   trayectoria solo aporta in-domain (0.103 vs 0.113).
4. **Mejor zero-shot entre aprendidos: posiciones-solo** (0.274) — aún por debajo del
   centroide sin entrenar (~0.23).

## ⚠️ Corrección al relato del v1

El contraste v0-no-transfiere / v1-sí-transfiere de `nivel-2-v1-temporal.md` **no
replica** con pre-entreno igualado (2000 pasos) y eval de 2 partidos: aquí el snapshot
v0 también muestra ventaja clara de fine-tune. La diferencia original probablemente se
debía al presupuesto de pre-entreno menor del v0 (~830 pasos) y al eval de 1 partido.
**Titular corregido del Nivel 2:** lo que el pre-entreno transfiere entre deportes es
el *uso de las features de velocidad* (difícil de aprender con pocos datos y necesitado
de recalibración), no la profundidad temporal per se. Las posiciones llevan la señal
central pero su mapping es fácil: 30 min desde cero bastan para aprenderlo.

## Estado de hipótesis del diseño original (cierre)

- H1 (transferencia zero-shot sustancial): **refutada** — el zero-shot aprendido nunca
  bate al centroide sin entrenar.
- H2 (velocidades > posiciones; orientación lo más transferible): **refutada** en su
  primera parte (posiciones ≫ velocidades como señal); irónicamente las velocidades sí
  son la parte *transferible del aprendizaje*. Orientación: no testeable sin pose.
- H3 (few-shot recupera el gap): parcialmente — la recuperación la hacen los datos del
  deporte destino; el pre-entreno añade un plus pequeño (~4-7%) vía velocidades.
