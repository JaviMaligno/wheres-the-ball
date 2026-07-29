# Related Work + abstract reframe — workflow proposal & citation triage (2026-07-29)

Source: `paper-relatedwork-reframe` workflow (6 agents, web-verified). Full raw proposal
(BibTeX + rewritten Related Work + abstract reframe + inline definitions) archived in the
run output. **NOT yet applied to the paper** — pending (a) citation vetting below, and
(b) the B2/B4 reruns (the Related Work/abstract quote deep-vs-geo and transfer numbers
that will change once the tuned deep + unified-protocol transfer land).

## Citation triage (rigor: do not paste anything with placeholder authors)

### SOLID — real, canonical, correct IDs. Safe to add.
- `maksai2016players` — Maksai, Wang, Fua, CVPR 2016 (arXiv:1511.06181). **Core prior art.**
- `kim2023ballradar` — BallRadar, KDD 2023 (arXiv:2306.08206), real title/authors + code repo. **Closest prior work.**
- `capellera2024transportmer` — TranSPORTmer, ACCV 2024 (arXiv:2410.17785). **Deep ceiling candidate.**
- `spearman2018beyond` — Spearman, Sloan 2018. Pitch-control/space value.
- `spearman2017physics` — Spearman et al., Sloan 2017. Pitch control.
- `fernandez2018wide` — Fernández & Bornn, Sloan 2018. Space value.
- `kamath2023whatsup` — What'sUp, EMNLP 2023 (arXiv:2310.19785). VLM spatial failure.
- `fu2024blink` — BLINK, ECCV 2024 (arXiv:2404.12390). VLM perception.
- `xia2025sportu` — SPORTU (arXiv:2410.08474). Sports-VLM benchmark. (verify author list before camera-ready)

### QUARANTINE — placeholder authors / unverifiable future IDs. Do NOT paste; verify individually or drop.
- `kim2026pathcrf` (arXiv:2602.12080, 2026) — plausible but author list unconfirmed; future ID.
- `jia2026omnispatial` (arXiv:2506.03135) — key/year mismatch, "and others".
- `xia2026sportr` (arXiv:2511.06499, 2026) — "and others".
- `courtsi2026` — author `{Visionary Laboratory}` (corporate placeholder) — RED FLAG.
- `soccerlens2026` — author `Anonymous` — NOT verified.
- `sportd2026` — author `Anonymous` — NOT verified.

Decision: the SOLID set already covers every mandatory reviewer demand (ball-from-players
prior art + pitch control + canonical VLM-spatial). The QUARANTINE set is
nice-to-have differentiation only; drop unless each can be individually confirmed real
(real authors, resolvable arXiv). Rewrite the drafted Related Work to cite only the SOLID
set (the current draft cites the quarantined keys — must be trimmed).

## .bst caveat (flagged by the synthesize agent, real)
The new VLM entries use `eprint/archivePrefix`; the existing refs.bib uses `note={arXiv:...}`.
The paper's `\bibliographystyle{plain}` does NOT print `eprint` — arXiv ids would silently
vanish. Convert all arXiv ids to `note={arXiv:XXXX.XXXXX}` for consistency with the plain style.

## Apply order
1. After B2 (tuned deep) + B4 (unified transfer) land → finalize the numbers in abstract/body.
2. Trim the drafted Related Work to the SOLID citations, convert arXiv ids to `note=`, add to refs.bib.
3. Apply abstract reframe (lead with result) + inline definitions (coupling / leak control / win-rate).
4. Recompile both 1-col and 2-col; verify all `\cite` resolve.
