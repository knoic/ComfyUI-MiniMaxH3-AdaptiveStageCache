import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from adaptive import ControllerConfig, SelfCalibratingController


def test_alignment_blocks_reuse_until_observed():
    c = SelfCalibratingController(ControllerConfig(alignment_steps=2))
    assert not c.should_reuse(0.001)
    c.observe(0.1, 0.01)
    assert not c.ready
    c.observe(0.1, 0.01)
    assert c.ready


def test_budget_and_refresh():
    c = SelfCalibratingController(ControllerConfig(error_budget=0.05, alignment_steps=1, process_noise=0, measurement_noise=0.01, uncertainty_weight=0))
    c.observe(0.1, 0.01)
    assert c.should_reuse(0.1)
    c.record_reuse(0.1)
    assert c.should_reuse(0.1)
    c.record_reuse(0.1)
    assert not c.should_reuse(0.1)
    c.observe(0.1, 0.01)
    assert c.accumulated_error == 0


def test_invalid_input_never_reuses():
    c = SelfCalibratingController(ControllerConfig(alignment_steps=1))
    c.observe(0.1, 0.1)
    assert not c.should_reuse(math.nan)


def test_uncertainty_is_scaled_into_output_change_units():
    c = SelfCalibratingController(ControllerConfig(error_budget=0.055, alignment_steps=1, process_noise=0.004, measurement_noise=0.012))
    c.observe(0.1, 0.01)
    # A small latent change should be eligible.  The uncertainty is a ratio
    # and must be scaled by this input change before comparison with budget.
    assert c.should_reuse(0.01)


def test_sensitivity_gate_can_veto_a_ready_cache():
    c = SelfCalibratingController(ControllerConfig(alignment_steps=1, sensitivity_budget=0.01))
    c.observe(0.1, 0.01)
    assert c.should_reuse(0.01, sensitivity_error=0.005)
    assert not c.should_reuse(0.01, sensitivity_error=0.02)
