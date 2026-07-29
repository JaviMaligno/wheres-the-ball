# Paper review synthesis & strengthening plan (2026-07-29)

Five independent adversarial reviews (statistical rigor, fact-check, clarity, novelty/venue,
reproducibility) of `paper/`. **Aggregate verdict: reject / major-revision in current form,
recoverable.** The empirical spine is real and reproduces from the JSONs (geo 0.099±0.006 vs
deep 0.126±0.009 vs centroid 0.213, 12/12; velocity ablation +0.035, 12/12; TDA adds nothing;
calibration sign-contradiction). What sinks it as submitted: missing prior-art positioning, a
strawman deep baseline, an unmeasured VLM↔specialist bridge, a transfer protocol/metric
inconsistency, and missing leak/inpainting reproducibility — plus factual cleanups.

**Strategy (Javier): close each finding by doing the missing work, not by excusing it in prose.**

## BLOCKERS — real work

- **B1. Prior art.** Add and *position against*: Maksai (CVPR 2016), Kim/BallRadar (KDD 2023,
  arXiv:2306.08206 — near-identical task + permutation-invariant set model), Capellera/TranSPORTmer
  (ACCV 2024), pitch control (Spearman, Sloan 2018). `docs/00-vision-general.md` already mandated
  citing Maksai/Kim/Capellera. Rewrite Related Work; differentiate our angle = counterfactual
  (object removed) + de-biased + minimal-signal dissection + calibration. [writing; no compute]
- **B2. DONE (2026-07-29).** Tuned multi-seed DeepSets (val split + early stopping, 3 seeds),
  LOMO over 12 matches (`paper_deep_tuned.py`): tuned deep **0.087±0.008** (across-seed std 0.002 →
  stable, also closes M10 for the deep) beats geo 0.099 in 12/12. **Corrected recovery = 91%±3%**
  (was the strawman 132%). Fig 2 regenerated + §4.2/caption/abstract rewritten to the honest
  "geometry recovers 91% of a tuned deep model; deep keeps a small edge" framing. Compiles clean.
- **B3. Measure the VLM↔specialist bridge.** Fig 1 implies the specialist beats the VLMs but they
  never competed on the same items/input (specialist = field GT tracks; VLMs = SoccerNet pixels).
  Run a head-to-head on a shared set: geometry model on the n=260 `players` field, and/or VLMs on
  the same 1s tracks (`nivel1_vlm_tracks.py` exists). [local ~$0 for specialist; VLM-on-tracks adds API $]
- **B4. Cross-sport transfer: one protocol, one metric.** Deep transfer (0.196/0.395) uses random
  80/20 (leakage-prone: adjacent frames in train+test); geo uses LOMO. And "0.52→0.09" is
  untraceable (deep-check ratio 0.52→0.09; geo `asym_ratio` −0.75→−0.08; decompose 0.16→0.09 —
  three different pairs). Rerun deep transfer under LOMO, pick ONE asymmetry metric, report both
  models on it, state which model each number is. [local, multi-seed, ~$0, ~1h]
- **B5. DONE (2026-07-29).** Ran the VLM leak check on all 260 items (0 errors). Flag rate
  **33/260 = 12.7%** (ball or artifact judged visible). Re-ran the far-bin comparison excluding
  flagged items — **headline robust**: GPT far 52%→54% (median 0.354→0.345), Llama 34%→32%
  (0.454→0.453). No VLM beats the camera baseline on off-center balls with or without leak-flagged
  items. Leak field now persisted in predictions.json. → add the flag rate + the excl-leak robustness
  row to the paper.
- **B6. Fix inpainting reproducibility.** Results use `masked/*.png` but `inpaint_lama.py` writes
  `.jpg`, and `uvx --from iopaint` floats the version. Reconcile which script produced the pngs,
  pin the iopaint/LaMa version + settings (device, checkpoint, resize), document. [local ~15min]
- **B7. Complete the Claude runs.** Opus 96/260, Sonnet 69/260 (Sonnet 0 far items) — partial due
  to Anthropic credit exhaustion. NEEDS credit recharge (Javier), then finish. [needs credit; ~$15, ~2–3h]

## FACTUAL ERRORS — fix regardless (quick, no compute)

- **F1.** §4.6 "positions-only 0.258 vs 0.111" — 0.258 is **TDA-only**, not positions-only.
- **F2.** "error highest when still 12/12" → **10/12** (12/12 is only the velocity-contribution claim).
- **F3.** RESOLVED (labeling bug, data is consistent): soccer = **249,917** frames (~250k, 12 matches),
  basketball = **231,351** frames (~231k, 6 games). The paper's "231k" is *basketball alone*, wrongly
  presented as the cross-sport/combined count. Fix: say "~250k soccer + ~231k basketball (~481k total)",
  not "231k soccer+basket".
- **F4.** "ten features" → **11** named features (or hedge "~ten").
- **F5.** Fig 1 caption "n=120" — Llama far bin is **118**.
- **F6.** "court ~1/6 the area" → actually **~1/16** (28×15 vs 105×68); keep the ~4× linear/velocity.
- **F7.** "0.52→0.09" transfer ratio (overlaps B4) — make traceable to one model/definition.
- **F8.** `nivel3_informational.py` hard-codes "+0.20"/"−0.17" as print strings; print computed values.

## MAJORS — framing + some work

- **M1.** Abstract: lead with the result (VLMs fail, 60 kB model succeeds), not the throat-clearing setup.
- **M2.** Define inline: *coupling*, *leak control*, *win-rate*, the transfer *ratio*.
- **M3.** §4.4–4.6 need a table or figure (the calibration finding deserves a Fig 4 more than the vel ablation).
- **M4.** "GPT at chance" is failure-to-reject dressed as finding; its far **median 0.354 < center 0.363**
  (GPT slightly *beats* center on median). Reframe honestly / acknowledge underpower (far n=120).
- **M5.** Direction result is **n=2** (g1↔g2) and ~coin-flip (within-45°=50%); run it on the 12 matches
  or soften. [local ~$0]
- **M6.** Add **pitch control** as a run, cited geometric baseline (the field's standard "space toward ball"). [local]
- **M7.** Differentiate the VLM half against the 2024–26 sports-VLM benchmarks (SPORTU, SoccerLens,
  "Stepping VLMs onto the Court", SportD) — the counterfactual + camera-debias is the novel axis.
- **M8.** Venue spine: keep at CVsports, **lead with the tracking/geometry half**, VLM half as contrast.
- **M9.** Add code/data-availability + ethics statements; per-dataset license nuance (SportVU = no
  explicit license, research-only, scripts-not-data; SoccerNet/Metrica/SkillCorner each their own terms).
- **M10.** Multi-seed for learned models (DeepSets, MDN); state plotted ±std is inter-match, not inter-seed.
  [folds into B2/B4]
- **M11.** De-biasing is reweighting, not a causal control; add a permutation / label-shuffle control
  (predict a random other frame's ball) to bound residual structure. [local ~$0]
- **M12.** Broadcast-camera confound: "off-center ball" entangled with "unusual framing" in the far bin — acknowledge.

## MINORS

- Title "60-Kilobyte" reads blog-ish; params are ~12.9k / ~51.7 kB (not 14k/60 kB) — round down or be exact.
- Terminology: unify Deep Sets / specialist / black box; keep *still* (speed) vs *off-center/loose*
  (distance-to-mass) as **two distinct axes** (§4.3 is still-axis, §4.5 is off-center-axis).
- Drop "pre-registered" for the TDA rule (no timestamped registration) or register it.
- Reproducibility appendix: hyperparams (d_h, steps, K epochs, GBM max_iter), feature table (all
  20 cols + `converge`/`team_contact` defs, SPEED_MIN), bin & still thresholds, VLM decoding params
  (temp/max_tokens/seed) + model snapshot dates.
- Citation hygiene: `ripser` bib year (2018 vs key 2021); add model cards for GPT-5.4/Llama-4/Claude.
- Multiple comparisons across bins×models×features with no family-wise correction — note it.

## Proposed execution order

1. **Factual fixes F1–F8** (minutes, no compute) — stop the bleeding.
2. **B5 leak + B6 inpainting** (cheap, unblock reproducibility of the benchmark).
3. **B2 tuned/multi-seed deep + B4 transfer LOMO** (one modeling push; kills the two stat blockers).
4. **B3 bridge** (specialist on n=260 / VLM-on-tracks) — the central rhetorical claim.
5. **M5 direction @12, M6 pitch control, M11 shuffle control** (local, round out rigor).
6. **B7 Claude** (once credit recharged).
7. **B1 related work + M1/M2/M3/M7/M8/M9 rewrite** + minors.

Note: the earlier injection anomaly in one review run was NOT from any repo file (the repro reviewer
explicitly scanned and found no suspicious content); the codebase is clean.
