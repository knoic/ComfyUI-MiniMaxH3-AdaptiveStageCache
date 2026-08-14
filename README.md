# ComfyUI MiniMax H3 AdaptiveStageCache

> **Experimental — native ComfyUI MiniMax H3 only.** This project is an
> independent, training-free research implementation. It has unit-tested
> controller logic but has **not yet been benchmarked on an H3 checkpoint**.
> Do not use it for production work without fixed-seed A/B review.

AdaptiveStageCache is a MiniMax H3 acceleration node designed around a simple
principle: preserve a small live part of the transformer as a current-step
probe, while reusing only those later transformer-stage residuals whose
predicted drift remains inside a per-stage error budget.

It is not a drop-in copy or port of NaviCache or BWCache. The project borrows
research ideas from both, then adapts the cache boundary and safety checks to
the native ComfyUI `MiniMaxH3Model` patch API.

## What it does

For a stage transformation `G_s` and its input `h_s`, a full execution stores:

```text
residual_s = G_s(h_s) - h_s
```

On an approved cache hit it returns:

```text
G_s(h_s,current) ≈ h_s,current + residual_s,previous_full
```

The first stage always executes. The remaining H3 blocks are split into a small
number of contiguous stages (two by default: one live probe plus one cacheable
tail), so the node can refresh one
stage independently of another. This is intentionally different from a full
model residual cache, which skips every transformer block at once.

Each cacheable stage has a lightweight self-calibrating controller. On full
executions it observes a local sensitivity estimate:

```text
ratio = relative_output_change / relative_input_change
```

It smooths that estimate with a scalar Kalman-style update, grows uncertainty
while reusing a cached residual, and forces an exact execution when the
predicted accumulated error exceeds the stage error budget. The implementation
also has these hard safety rails:

- initial full executions for stage alignment;
- a maximum chain of cached hits;
- cache disabled in the first 12% and final 12% of denoising by default;
- reset on shape, dtype, device or reversed-sigma changes;
- separate contexts for ComfyUI condition UUIDs;
- optional maximum-per-frame target-video latent guard where H3 layout metadata
  is available;
- a SenCache-inspired sensitivity veto: after live observations, reuse must
  satisfy both the existing accumulated-error controller and an online estimate
  of latent-change plus sigma-change sensitivity;
- rejection of EasyCache, CacheDiT, T8 Block Cache and other `double_block`
  patch replacements.

### SenCache adaptation boundary

SenCache's released implementation uses offline, per-timestep Jacobian norms
precomputed for Wan, CogVideoX and LTX models. Those tables are neither
available for MiniMax H3 nor transferable across architectures. This node does
**not** load or pretend to reproduce those weights. Instead, it learns a
per-stage scalar sigma sensitivity only from fresh H3 stage executions and
uses it as a conservative second gate:

```text
estimated_sensitivity_error = local_latent_sensitivity * Δlatent
                              + sigma_weight * learned_sigma_sensitivity * Δsigma
```

If that error exceeds `sensitivity_budget`, the stage runs exactly. The guard
stores scalars only, so it adds no persistent GPU activation tensor beyond the
existing residual and stage-input cache. It is an H3-specific
**SenCache-inspired** safety mechanism, not an official SenCache port.

## Why stages instead of a whole-model cache?

The official NaviCache code caches an entire model residual. Its conditional and
unconditional CFG forwards share a skip decision but retain separate residuals.
The official BWCache implementation uses block-level L1 statistics to schedule
reuse, but when reuse is selected it also applies one previously cached
transformer residual. Neither repository currently supports MiniMax H3.

For H3, a whole-model hit is cheap to implement but has a large failure domain:
a local motion or audio-token change can make every skipped block stale. This
project therefore treats the official implementations as **baselines to beat**,
not as code to transplant. A later benchmark should compare all three cache
boundaries under exactly the same H3 workflow:

| Variant | Reused quantity | Intended use |
|---|---|---|
| Whole-model baseline | `F(x) - x` | Establish NaviCache-style speed/quality baseline |
| FirstBlockCache | tail residual after live block 0 | Low-memory conservative baseline |
| AdaptiveStageCache | residual for each later H3 stage | Main experimental route |

## Install

Clone this repository into `ComfyUI/custom_nodes`, then restart ComfyUI:

```bash
git clone <repository-url>
```

Place **MiniMax H3 AdaptiveStageCache (Experimental)** immediately after
**Load Diffusion Model**. Connect its `MODEL` output everywhere the original
model output was used, including the scheduler and guider.

Do not connect it together with another H3 block-replacement or cache node.

## Modes

- **Safe — self-calibrating:** one consecutive cached hit and the lowest error
  budget.
- **Balanced — self-calibrating:** two consecutive hits; default.
- **Fast — self-calibrating:** three consecutive hits; inspect motion, identity,
  text and audio carefully.
- **Custom — experimental values:** exposes error budget, stage count,
  denoising window, hit limit, temporal guard and the sensitivity gate.

The numeric thresholds are provisional. They are not transferable from Wan,
HunyuanVideo, Open-Sora, EasyCache, NaviCache or BWCache, because their change
metrics, cache boundaries, sampler settings and model architectures differ.

## Validation protocol

Before calling a preset usable, run a fixed benchmark matrix:

1. Use the same H3 checkpoint, prompt, seed, scheduler, CFG, resolution,
   duration and step count for each row.
2. Record warm and cold wall time, peak VRAM, model execution time, cache hit
   count and per-stage hit count.
3. Review decoded video for identity drift, local-motion failure, flicker,
   typography errors and audio/video desynchronization.
4. Compare decoded output with uncached output using LPIPS/SSIM/PSNR only as
   trajectory-distance signals; do not treat them as perceptual quality alone.
5. Keep the first configuration that has no material visual regression, rather
   than selecting the largest reported speedup.

## Current status

- Controller unit tests are included in `tests/test_adaptive.py`.
- Syntax checks and controller tests pass in the development environment.
- A full ComfyUI + H3 integration test and benchmark are still required.
- No custom CUDA/Triton kernel is included; this project only uses ComfyUI model
  patching and PyTorch tensor operations.

## Research and implementation references

The following materials informed the design. Their code was **not copied** into
this repository.

1. Lv et al., [NaviCache: Test-Time Self-Calibration Caching for Video
   Generation](https://arxiv.org/abs/2606.26795), ICML 2026. Used for the
   initial-alignment, state-ratio, process/measurement-noise and accumulated
   predicted-error concepts.
2. [NaviCache official implementation](https://github.com/HelloZicky/NaviCache)
   (Apache-2.0). Reviewed to verify that its released code implements a
   whole-model residual cache and condition-pair logic, not a block cache.
3. Cui et al., [BWCache: Accelerating Video Diffusion Transformers through
   Block-Wise Caching](https://arxiv.org/abs/2509.13789), ICLR 2026. Used for
   the mid-denoising redundancy observation, relative-L1 signal, periodic
   refresh and final-refinement protection.
4. [BWCache official implementation](https://github.com/hsc113/BWCache) (MIT).
   Reviewed to verify its released global reuse schedule and full-transformer
   residual application.
5. Li et al., [Sol Video Inference Engine](https://arxiv.org/abs/2606.23743),
   2026. Used for the model × hardware × workflow-specific benchmark philosophy.
6. DuckyShell, [ComfyUI MiniMax H3 FirstBlockCache](https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache)
   (MIT). Used only as a ComfyUI-native H3 patching and cache-lifecycle
   compatibility reference.
7. [SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware
   Caching](https://arxiv.org/abs/2602.24208), CVPR 2026, and the
   [official implementation](https://github.com/vita-epfl/SenCache). Used to verify the Jacobian-weighted
   `J_z * Δz + J_t * Δt` reuse-gate form; this project replaces unavailable H3
   offline Jacobians with online stage-local scalar estimates.

If this project reports research-comparable results, it should cite the papers
above in addition to clearly identifying this implementation as independent.

## License

MIT. See [LICENSE](LICENSE).
