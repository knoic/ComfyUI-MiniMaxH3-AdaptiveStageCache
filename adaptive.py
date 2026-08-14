from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

def relative_l1(current, previous) -> float:
    """A scale-invariant change measurement with a finite zero safeguard."""
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean().clamp(min=1e-8)
    return float((numerator / denominator).item())


@dataclass(frozen=True)
class ControllerConfig:
    error_budget: float = 0.055
    alignment_steps: int = 2
    max_consecutive_hits: int = 2
    process_noise: float = 0.004
    measurement_noise: float = 0.012
    uncertainty_weight: float = 1.0
    sensitivity_budget: float = 0.080


@dataclass
class SelfCalibratingController:
    """A small scalar Kalman-style controller inspired by NaviCache.

    It estimates the local output/input change ratio for a *single stage*.  The
    controller intentionally never fabricates an observation: it only updates
    its state after a real stage execution.
    """

    config: ControllerConfig
    ratio: Optional[float] = None
    variance: float = 0.0
    accumulated_error: float = 0.0
    observations: int = 0
    consecutive_hits: int = 0

    def reset(self) -> None:
        self.ratio = None
        self.variance = 0.0
        self.accumulated_error = 0.0
        self.observations = 0
        self.consecutive_hits = 0

    @property
    def ready(self) -> bool:
        return self.ratio is not None and self.observations >= self.config.alignment_steps

    def should_reuse(
        self,
        input_change: float,
        external_guard: bool = True,
        sensitivity_error: Optional[float] = None,
    ) -> bool:
        if not self.ready or not external_guard or not math.isfinite(input_change):
            return False
        if sensitivity_error is not None and (
            not math.isfinite(sensitivity_error)
            or sensitivity_error >= self.config.sensitivity_budget
        ):
            return False
        if self.consecutive_hits >= self.config.max_consecutive_hits:
            return False
        predicted_change = max(self.ratio, 0.0) * input_change
        # ``variance`` is in sensitivity-ratio units, while ``error_budget``
        # is in output-change units.  Convert it through the current input
        # change before applying the safety margin.  Adding sqrt(variance)
        # directly made the default budget impossible to satisfy, so no cache
        # hit could occur even after a successful alignment.
        predicted_uncertainty = math.sqrt(max(self.variance + self.config.process_noise, 0.0)) * input_change
        proposed_error = self.accumulated_error + predicted_change + self.config.uncertainty_weight * predicted_uncertainty
        return math.isfinite(proposed_error) and proposed_error < self.config.error_budget

    def record_reuse(self, input_change: float) -> None:
        self.variance += self.config.process_noise
        self.accumulated_error += max(self.ratio or 0.0, 0.0) * input_change
        self.consecutive_hits += 1

    def observe(self, input_change: Optional[float], output_change: Optional[float]) -> None:
        """Correct the ratio estimate from a freshly executed stage."""
        self.consecutive_hits = 0
        self.accumulated_error = 0.0
        if input_change is None or output_change is None or input_change <= 1e-8:
            return
        measured_ratio = output_change / input_change
        if not math.isfinite(measured_ratio):
            return
        if self.ratio is None:
            self.ratio = measured_ratio
            self.variance = self.config.measurement_noise
        else:
            prior_variance = self.variance + self.config.process_noise
            gain = prior_variance / (prior_variance + self.config.measurement_noise)
            self.ratio = (1.0 - gain) * self.ratio + gain * measured_ratio
            self.variance = (1.0 - gain) * prior_variance
        self.observations += 1
