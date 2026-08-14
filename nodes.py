from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import comfy.patcher_extension

from .adaptive import ControllerConfig, SelfCalibratingController, relative_l1


@dataclass(frozen=True)
class CacheConfig:
    controller: ControllerConfig
    start_percent: float = 0.12
    end_percent: float = 0.88
    stage_count: int = 2
    temporal_guard: bool = True
    temporal_limit: float = 0.11
    sensitivity_guard: bool = True
    sigma_sensitivity_weight: float = 0.25
    sea_filter: bool = True


PRESETS = {
    "Safe — self-calibrating": CacheConfig(ControllerConfig(error_budget=0.040, max_consecutive_hits=1)),
    "Balanced — self-calibrating": CacheConfig(ControllerConfig(error_budget=0.055, max_consecutive_hits=2)),
    "Fast — self-calibrating": CacheConfig(ControllerConfig(error_budget=0.075, max_consecutive_hits=3)),
}
CUSTOM_MODE = "Custom — experimental values"


@dataclass
class StageState:
    controller: SelfCalibratingController
    cached_residual: torch.Tensor | None = None
    previous_input: torch.Tensor | None = None
    pending_input: torch.Tensor | None = None
    previous_sigma: float | None = None
    previous_input_sigma: float | None = None
    pending_sigma: float | None = None
    sigma_sensitivity: float | None = None
    use_cache: bool = False
    hits: int = 0
    full_steps: int = 0

    def clear(self) -> None:
        self.cached_residual = None
        self.previous_input = None
        self.pending_input = None
        self.previous_sigma = None
        self.previous_input_sigma = None
        self.pending_sigma = None
        self.sigma_sensitivity = None
        self.use_cache = False
        self.controller.reset()
        self.hits = 0
        self.full_steps = 0


@dataclass
class RunContext:
    stages: list[StageState]
    previous_sigma: float | None = None
    input_signature: tuple | None = None
    video_slice: tuple[int, int] | None = None
    latent_frames: int | None = None

    def clear(self) -> None:
        for stage in self.stages:
            stage.clear()
        self.previous_sigma = None
        self.input_signature = None
        self.video_slice = None
        self.latent_frames = None


class AdaptiveStageCache:
    def __init__(self, config: CacheConfig, start_sigma: float, end_sigma: float, block_count: int):
        self.config, self.start_sigma, self.end_sigma = config, start_sigma, end_sigma
        self.ranges = self._make_ranges(block_count, config.stage_count)
        self.contexts: dict[tuple, RunContext] = {}
        self.current: RunContext | None = None
        self.full_calls = 0
        self.cached_stages = 0

    @staticmethod
    def _make_ranges(block_count: int, requested_stages: int) -> list[tuple[int, int]]:
        # Stage 0 is a live probe and is never skipped.  The tail is split evenly.
        tails = max(1, min(requested_stages - 1, block_count - 1))
        ranges = [(0, 0)]
        remaining, start = block_count - 1, 1
        for group in range(tails):
            size = math.ceil(remaining / (tails - group))
            ranges.append((start, start + size - 1))
            start += size
            remaining -= size
        return ranges

    @staticmethod
    def _signature(x):
        tensors = x if isinstance(x, (list, tuple)) else (x,)
        return tuple((tuple(t.shape), t.dtype, t.device) for t in tensors if torch.is_tensor(t))

    @staticmethod
    def _video_layout(payload):
        layout = payload.get("layout") if payload else None
        if layout is None:
            return None, None
        video_slice = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
        frames = layout.signature[1] if len(layout.signature) > 1 else None
        return video_slice, frames

    def _new_context(self) -> RunContext:
        return RunContext([StageState(SelfCalibratingController(self.config.controller)) for _ in self.ranges])

    def reset(self) -> None:
        for context in self.contexts.values():
            context.clear()
        self.contexts.clear()
        self.current = None
        self.full_calls = self.cached_stages = 0

    def begin_call(self, x, timestep, transformer_options, minimax_payload=None) -> None:
        sigma = float(timestep.flatten()[0].item()) / 1000.0
        uuids = transformer_options.get("uuids") or ("default",)
        key = tuple(str(value) for value in uuids)
        context = self.contexts.setdefault(key, self._new_context())
        signature = self._signature(x)
        if context.input_signature != signature or (context.previous_sigma is not None and sigma > context.previous_sigma + 1e-7):
            context.clear()
        context.input_signature, context.previous_sigma = signature, sigma
        context.video_slice, context.latent_frames = self._video_layout(minimax_payload)
        self.current = context

    def end_call(self) -> None:
        self.current = None

    def in_window(self) -> bool:
        return self.current is not None and self.end_sigma <= self.current.previous_sigma <= self.start_sigma

    def _temporal_ok(self, current: torch.Tensor, previous: torch.Tensor) -> bool:
        context = self.current
        if not self.config.temporal_guard or context is None or context.video_slice is None or not context.latent_frames:
            return True
        start, stop = context.video_slice
        now, old = current[start:stop], previous[start:stop]
        if now.shape != old.shape or now.shape[0] % context.latent_frames:
            return True
        rows = now.shape[0] // context.latent_frames
        now, old = now.reshape(context.latent_frames, rows, -1), old.reshape(context.latent_frames, rows, -1)
        frame_change = (now - old).abs().mean(dim=(1, 2)) / old.abs().mean(dim=(1, 2)).clamp(min=1e-8)
        return float(frame_change.max().item()) <= self.config.temporal_limit

    @staticmethod
    def _sea_temporal_filter(features: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply SeaCache's scheduler-aligned Wiener form along time only.

        H3's public ComfyUI patch payload exposes a video-token range and frame
        count, but not a reliable 3-D latent grid.  Reducing each frame to its
        token mean lets us retain the temporal spectral component without
        inventing a spatial layout or materializing another activation cache.
        """
        count = features.shape[0]
        if count < 2:
            return features
        value = features.float()
        frequency = torch.fft.rfftfreq(count, device=value.device, dtype=torch.float32).abs()
        sigma = min(max(float(sigma), 1e-6), 1.0 - 1e-6)
        signal, noise = 1.0 - sigma, sigma
        spectrum_prior = 1.0 / frequency.square().clamp(min=1e-16)
        gain = signal * spectrum_prior / (signal * signal * spectrum_prior + noise * noise + 1e-16)
        gain = gain / gain.mean().clamp(min=1e-16)
        return torch.fft.irfft(torch.fft.rfft(value, dim=0) * gain[:, None], n=count, dim=0)

    def _cache_input_change(self, current: torch.Tensor, previous: torch.Tensor, current_sigma: float, previous_sigma: float | None) -> float:
        context = self.current
        if not self.config.sea_filter or context is None or context.video_slice is None or not context.latent_frames:
            return relative_l1(current, previous)
        start, stop = context.video_slice
        now, old = current[start:stop], previous[start:stop]
        if now.shape != old.shape or now.shape[0] % context.latent_frames:
            return relative_l1(current, previous)
        rows = now.shape[0] // context.latent_frames
        now = now.reshape(context.latent_frames, rows, -1).mean(dim=1)
        old = old.reshape(context.latent_frames, rows, -1).mean(dim=1)
        now = self._sea_temporal_filter(now, current_sigma)
        old = self._sea_temporal_filter(old, previous_sigma if previous_sigma is not None else current_sigma)
        return relative_l1(now, old)

    @staticmethod
    def _relative_sigma_change(current: float | None, previous: float | None) -> float:
        if current is None or previous is None:
            return 0.0
        return abs(current - previous) / max(abs(previous), 1e-8)

    def begin_stage(self, stage_index: int, x: torch.Tensor) -> bool:
        context = self.current
        if context is None:
            raise RuntimeError("AdaptiveStageCache called outside model execution")
        state = context.stages[stage_index]
        # The first stage is only a live probe.  Do not retain its feature
        # tensors, otherwise it silently consumes the same memory as a cached
        # stage while it is never eligible for reuse.
        if stage_index == 0:
            state.use_cache = False
            return False
        state.pending_input = x.detach()
        state.pending_sigma = context.previous_sigma
        state.use_cache = False
        if stage_index == 0 or not self.in_window() or state.cached_residual is None or state.previous_input is None:
            return False
        input_change = self._cache_input_change(
            x, state.previous_input, context.previous_sigma, state.previous_input_sigma
        )
        guard = self._temporal_ok(x, state.previous_input)
        sigma_change = self._relative_sigma_change(context.previous_sigma, state.previous_sigma)
        sensitivity_error = None
        if self.config.sensitivity_guard and state.sigma_sensitivity is not None:
            # SenCache gates reuse with J_z * Δz + J_t * Δt.  H3 has no
            # released offline Jacobian table, so estimate both local terms
            # online and keep this as a conservative veto, never a force-hit.
            latent_term = max(state.controller.ratio or 0.0, 0.0) * input_change
            sigma_term = self.config.sigma_sensitivity_weight * state.sigma_sensitivity * sigma_change
            sensitivity_error = latent_term + sigma_term
        if state.controller.should_reuse(input_change, guard, sensitivity_error):
            state.controller.record_reuse(input_change)
            state.previous_input = x.detach()
            state.previous_input_sigma = context.previous_sigma
            state.pending_input = None
            state.pending_sigma = None
            # The residual itself still originates at previous_sigma.  Keep
            # that anchor until a true full execution refreshes it.
            state.use_cache = True
            state.hits += 1
            self.cached_stages += 1
            return True
        return False

    def finish_full_stage(self, stage_index: int, output: torch.Tensor) -> None:
        context = self.current
        if context is None:
            raise RuntimeError("AdaptiveStageCache has no active context")
        state = context.stages[stage_index]
        if stage_index == 0:
            self.full_calls += 1
            return
        if state.pending_input is None:
            raise RuntimeError("AdaptiveStageCache lost the stage input")
        input_change = self._cache_input_change(
            state.pending_input, state.previous_input, state.pending_sigma, state.previous_input_sigma
        ) if state.previous_input is not None else None
        # Keep only residual + latest stage input persistently.  The previous
        # output is reconstructed for this full-step observation and is then
        # released, saving one full activation-sized tensor per cache stage.
        previous_output = (state.previous_input + state.cached_residual) if state.previous_input is not None and state.cached_residual is not None else None
        output_change = relative_l1(output, previous_output) if previous_output is not None else None
        state.controller.observe(input_change, output_change)
        sigma_change = self._relative_sigma_change(state.pending_sigma, state.previous_sigma)
        if output_change is not None and sigma_change > 1e-8:
            measured_sigma_sensitivity = output_change / sigma_change
            if math.isfinite(measured_sigma_sensitivity):
                state.sigma_sensitivity = measured_sigma_sensitivity if state.sigma_sensitivity is None else (
                    0.5 * state.sigma_sensitivity + 0.5 * measured_sigma_sensitivity
                )
        state.cached_residual = (output - state.pending_input).detach()
        state.previous_input = state.pending_input
        state.previous_input_sigma = state.pending_sigma
        state.pending_input = None
        state.previous_sigma = state.pending_sigma
        state.pending_sigma = None
        state.full_steps += 1
        self.full_calls += 1

    def finish_cached_stage(self, stage_index: int, x: torch.Tensor) -> torch.Tensor:
        state = self.current.stages[stage_index]
        if state.cached_residual is None:
            raise RuntimeError("AdaptiveStageCache has no residual for a cached stage")
        return x + state.cached_residual

    def summary(self) -> str:
        contexts = [self.current] if self.current is not None else list(self.contexts.values())
        persistent_tensors = 2 * sum(
            1
            for context in contexts
            for index, state in enumerate(context.stages)
            if index and state.cached_residual is not None and state.previous_input is not None
        )
        return f"cached stages {self.cached_stages}; full stage executions {self.full_calls}; stages {self.ranges}; persistent cache tensors {persistent_tensors}"


def make_block_patch(cache: AdaptiveStageCache, block_index: int, stage_index: int, first: int, last: int):
    def patch(args, extra):
        original = extra["original_block"]
        if block_index == first:
            cache.begin_stage(stage_index, args["img"])
        state = cache.current.stages[stage_index]
        if state.use_cache:
            if block_index == last:
                return {"img": cache.finish_cached_stage(stage_index, args["img"])}
            return {"img": args["img"]}
        output = original(args)["img"]
        if block_index == last:
            cache.finish_full_stage(stage_index, output)
        return {"img": output}
    return patch


def diffusion_wrapper(cache):
    def wrapper(executor, *args, **kwargs):
        options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        payload = args[4] if len(args) > 4 else kwargs.get("minimax_payload")
        cache.begin_call(args[0], args[1], options, payload)
        try:
            return executor(*args, **kwargs)
        finally:
            cache.end_call()
    return wrapper


def sample_wrapper(cache, label):
    def wrapper(executor, *args, **kwargs):
        cache.reset()
        logging.info("MiniMax H3 AdaptiveStageCache enabled: %s", label)
        try:
            return executor(*args, **kwargs)
        finally:
            logging.info("MiniMax H3 AdaptiveStageCache: %s", cache.summary())
            cache.reset()
    return wrapper


class ApplyMiniMaxH3AdaptiveStageCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": ([*PRESETS, CUSTOM_MODE], {"default": "Balanced — self-calibrating"}),
            "error_budget": ("FLOAT", {"default": 0.055, "min": 0.005, "max": 1.0, "step": 0.005}),
            "stage_count": ("INT", {"default": 2, "min": 2, "max": 16}),
            "start_percent": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_consecutive_hits": ("INT", {"default": 2, "min": 1, "max": 8}),
            "temporal_guard": ("BOOLEAN", {"default": True}),
            "sensitivity_guard": ("BOOLEAN", {"default": True}),
            "sensitivity_budget": ("FLOAT", {"default": 0.080, "min": 0.001, "max": 2.0, "step": 0.005}),
            "sigma_sensitivity_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            "sea_filter": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES, FUNCTION, CATEGORY = ("MODEL",), "apply", "MiniMax H3/optimization"
    DESCRIPTION = "Experimental stage-wise residual cache with SeaCache spectral distance, NaviCache calibration and a SenCache-inspired sensitivity veto."

    def apply(self, model, mode, error_budget, stage_count, start_percent, end_percent, max_consecutive_hits, temporal_guard, sensitivity_guard, sensitivity_budget, sigma_sensitivity_weight, sea_filter):
        if mode == CUSTOM_MODE:
            if start_percent >= end_percent:
                raise ValueError("start_percent must be smaller than end_percent")
            config = CacheConfig(
                controller=ControllerConfig(
                    error_budget=error_budget,
                    max_consecutive_hits=max_consecutive_hits,
                    sensitivity_budget=sensitivity_budget,
                ),
                start_percent=start_percent,
                end_percent=end_percent,
                stage_count=stage_count,
                temporal_guard=temporal_guard,
                sensitivity_guard=sensitivity_guard,
                sigma_sensitivity_weight=sigma_sensitivity_weight,
                sea_filter=sea_filter,
            )
            label = f"Custom budget {error_budget:.3f}, stages {stage_count}, SEA filter {sea_filter}"
        else:
            config, label = PRESETS[mode], mode
        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or not hasattr(diffusion_model, "blocks"):
            raise ValueError("AdaptiveStageCache only supports native ComfyUI MiniMaxH3Model")
        block_count = len(diffusion_model.blocks)
        if block_count < 3:
            raise ValueError("AdaptiveStageCache requires at least three transformer blocks")
        options = model.model_options.get("transformer_options", {})
        if "easycache" in options or "cache_dit_turbo" in options or "minimax_h3_block_cache_t8" in options:
            raise ValueError("AdaptiveStageCache cannot be combined with another cache node")
        existing = options.get("patches_replace", {}).get("dit", {})
        if any(("double_block", index) in existing for index in range(block_count)):
            raise ValueError("AdaptiveStageCache requires an unpatched MiniMax H3 double_block stack")
        sampling = model.get_model_object("model_sampling")
        cache = AdaptiveStageCache(config, float(sampling.percent_to_sigma(config.start_percent)), float(sampling.percent_to_sigma(config.end_percent)), block_count)
        patched = model.clone()
        for stage_index, (first, last) in enumerate(cache.ranges):
            for index in range(first, last + 1):
                patched.set_model_patch_replace(make_block_patch(cache, index, stage_index, first, last), "dit", "double_block", index)
        key = f"minimax_h3_adaptive_stage_cache_{id(cache)}"
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key, diffusion_wrapper(cache))
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key, sample_wrapper(cache, label))
        return (patched,)


NODE_CLASS_MAPPINGS = {"ApplyMiniMaxH3AdaptiveStageCache": ApplyMiniMaxH3AdaptiveStageCache}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyMiniMaxH3AdaptiveStageCache": "MiniMax H3 AdaptiveStageCache (Experimental)"}
