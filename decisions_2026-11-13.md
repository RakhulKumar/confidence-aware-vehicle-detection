# Locked decisions (13 Nov 2026)

| # | Decision | Rationale |
|---|---|---|
| D1 | BMD-45 = development, AU-INV-PALS = transfer target (reverse of plan). | PALS measured limits: ~50 occluded vehicles in its labelled set, incomplete labels, ~5% of vehicles never proposed by BMD-trained detectors. To record the reverse direction honestly, v0.6 trains on PALS and tests on BMD-45 (no earlier valid run exists to cite). |
| D2 | Class-agnostic matching primary; native/coarse in appendix. | Occlusion misses are visibility failures; class confusions are analysed separately (RQ5). |
| D3 | Both UQ arms run now: 3-seed deep ensemble AND evidential head. Nothing left as future work. | Evidential is the plan's secondary UQ and gives RQ5 a direct aleatoric/epistemic test instead of the post-hoc proxy. |
| D4 | Retrain on the full BMD-45 train role (splits unchanged). | Reviewers judge the method on competitive detectors; one final retrain covers seeds + evidential so all numbers are produced once. |
| D5 | RQ2 wording unchanged; its answer framed as "certified object-level recall, with occlusion-conditional coverage that brings occluded vehicles up to the target at a stated flag cost". | Recall-per-false-alarm gain is mostly duplicate suppression = Gilg et al. 2023 (prior art). The conditional guarantee is the novel part. |
| D6 | Primary 1-alpha = 0.90; 0.95 and full alpha curve also reported. | alpha=0.1 is the standard in conformal detection (Andeol 2023/2025, Hulsman 2024, Li 2022), so baselines compare directly; 0.90 costs ~1.1-1.4 flags/img vs 4-6 at 0.95. |
| D7 | Headline rule: occlusion-context Mondrian conformal; global miss-aware rule as reference. | Only element without detection prior art. Global rule meets 0.90 on average but gives the most-occluded vehicles ~0.82; Mondrian brings every stratum to target. Reallocation (unoccluded recall pays) is reported as the explicit trade-off. |
| D8 | Add baselines: per-image CRC on FNR (Hulsman/Andeol), SeqCRC (COD toolkit), PAC threshold (Li 2022), IoU-aware calibration (Gilg 2023), LaECE0 + LRP (Kuzucu 2024); soft-NMS if pre-NMS boxes available. | Pre-empt reviewer requests. Prior results we reproduce are cited as consistent with those papers. |
| D9 | Journal (Q1), full RQ1-RQ5. | |
| D10 | UA-DETRAC: occlusion-proxy validation against annotated occlusion + second transfer target. | Only candidate with per-vehicle occlusion annotation in fixed-camera traffic. |

No snapshot now (storage). Freeze once, after the final retrain and full rerun.
