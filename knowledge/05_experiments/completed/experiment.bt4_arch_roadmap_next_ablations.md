---
id: experiment.bt4_arch_roadmap_next_ablations
type: experiment
title: Experiment - BT4 architecture roadmap next ablations
status: completed
created: 2026-05-11
updated: 2026-05-12
priority: medium
depends_on:
  - "[[Design - SquareFormer-AV-PUCT]]"
  - "[[Design - 100M h7-h8 BT4 training pipeline]]"
related:
  - "[[Finding - Chess-specific geometry is high ROI for tiny models]]"
  - "[[Concept - SquareFormer]]"
source_notes:
  - docs/bt4_chessformer_browser_architecture_synthesis.md
  - docs/transformer_model_roadmap.md
agent_summary: >
  Historical/source note for the BT4/SquareFormer 10M ablation push after board-flipping became standard. The lane is frozen for now: bt4_h2_flip_av_relbank_d256_l8 is the provisional winner for LC0 promotion, and further ablations require explicit reopen criteria from the LC0 architecture funnel decision.
---

# Experiment - BT4 Architecture Roadmap Next Ablations

> Status: completed/frozen for now. Do not launch more variants from this list unless [[Decision - LC0 architecture funnel and deployability frontier]] reopen criteria are met.

## Outcome

Provisional BT4/SquareFormer 10M winner for promotion:

```text
bt4_h2_flip_av_relbank_d256_l8
```

Quick supervised metrics from the completed 10M roadmap run:

```text
dev_policy_ce=1.623149
dev_wdl_ce=0.924964
dev_policy_top1=0.483540
dev_policy_top4=0.829080
dev_policy_top8=0.937780
dev_av_mse=0.024651
composite=1.903692
```

This experiment remains useful as source context for ideas that may be reopened later, but it is no longer an active launch queue.

## Current baseline context

The active BT4/SquareFormer 10M flipped lane already covers:

- h2 compact square tokens with `board_normalization=stm_white_rankflip_v1`
- static relation bias / template-bank relation bias
- separate action-value, rank, and regret heads
- dynamic pooled relation gate for GAB-lite-style template weighting
- d128 and d192 small-model scale checks

The current implementation does **not** yet cover every architecture idea from the BT4/Chessformer synthesis.

## Was ready to run before freeze

These variants were part of the 10M ablation question and should now be treated as archived/source context, not a live queue:

1. `d128, 8 heads, template_bank`
   - Tests whether smaller head dimension with more heads helps chess geometry at the same hidden width.

2. `d128, 8 heads, template_bank + dynamic relation gate`
   - Tests whether the pooled board gate still helps when attention has more heads.

3. `d192, 8 heads, template_bank + dynamic relation gate`
   - Complements the existing d192 static/relbank run and asks whether dynamic geometry scales with width.

4. `d256, 8 layers, 8 heads, d_ff=512, template_bank`
   - First medium MiniBT scaling test. This is not browser-first; it measures whether transformer scale is buying strength before optimizing deployment.

All ready-to-run jobs must use flipped caches and export metadata with `board_normalization=stm_white_rankflip_v1`.

## Needs cache work first

The most important missing combined test is:

```text
h8 history
+ board flipping
+ template-bank or GAB-lite dynamic relation gate
+ AV/rank/regret heads
```

This should not be faked by mixing h2 AV/value overlays with h8 policy caches unless the stream and cache manifests explicitly declare compatible history and normalization.

Current status:

- 100M flipped h7/h8 SquareFormer policy caches have been submitted to AWS Batch.
- Matching h7/h8 `stm_white_rankflip_v1` value overlays exist locally.
- Matching h7/h8 `stm_white_rankflip_v1` ChessBench top-8 AV overlays exist locally.
- Training should wait for cloud cache manifests to finalize and validate.

## Needs trainer/export implementation first

These are architecture ideas from the roadmap that are not yet first-class in the current SquareFormer V2 trainer/export stack:

- True Chessformer-style GAB: pooled board state -> d2/d3 bottleneck -> per-layer/head `64x64` dynamic attention bias.
- Value-bucket head: 32 or 64 bins, separate from scalar Q and WDL.
- Uncertainty/value-error head: labels from teacher disagreement, value swing, or realized regret.
- Moves-left / horizon head: low-weight auxiliary only after labels are reliable.
- Global board token or BT4-style global-board embedding.
- Norm/activation/init ablations: Pre-LN vs Post-LN, GELU vs Mish, and DeepNet-style initialization.
- Side-relative token order: separate from board-flip normalization and high risk until policy/AV inverse mapping tests are exhaustive.

## Promotion rule

No added architecture complexity should graduate from this lane unless it improves supervised metrics and at least one play/search diagnostic. The useful target is not raw transformer novelty; it is better strength per byte, per FLOP, and per millisecond.

Current policy: use `bt4_h2_flip_av_relbank_d256_l8` for LC0 sanity/pilot/100M+ unless eval or correctness gates expose a blocker.
