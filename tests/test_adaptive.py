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
