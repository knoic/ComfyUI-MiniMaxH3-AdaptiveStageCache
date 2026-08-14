# ComfyUI MiniMax H3 AdaptiveStageCache

Experimental, training-free acceleration for native ComfyUI MiniMax H3.

This node combines two ideas:

- **stage-wise residual reuse:** block 0 remains live; the remaining H3 transformer blocks are divided into stages and a stage can reuse its most recently measured residual;
- **self-calibrating safety controller:** each stage tracks its input-to-output change ratio, uncertainty and accumulated error budget. It performs initial full-step alignment and forces refreshes when the error budget, uncertainty, sampling window or consecutive-hit limit says it is unsafe.

It is intentionally conservative and does not combine with EasyCache, CacheDiT, T8 Block Cache or any node that replaces H3 `double_block` patches.

## Install

Clone into `ComfyUI/custom_nodes`, then restart ComfyUI.

```bash
git clone <repository-url>
```

Connect **MiniMax H3 AdaptiveStageCache (Experimental)** directly after **Load Diffusion Model** and use its `MODEL` output for both scheduler and guider.

## Modes

- **Safe:** one consecutive cached hit; lowest budget.
- **Balanced:** two consecutive hits; default.
- **Fast:** three consecutive hits; review output carefully.
- **Custom:** exposes error budget, stage count, denoising window and hit limit.

The first stage always runs. Caching is disabled for the first 12% and last 12% of sampling. The optional temporal guard evaluates the most changed target-video latent frame where ComfyUI supplies H3 layout metadata.

## Status and validation

The controller unit tests are included. This package has **not yet been benchmarked on a real H3 checkpoint**, so its presets are safety-oriented starting points rather than quality claims. Use a fixed seed and compare against the uncached result before production work.
