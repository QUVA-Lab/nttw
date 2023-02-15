"""Tests for util_experiments.py"""
import numpy as np
from nttw.experiments import util_experiments


def test_dummy_inference():
  num_users = 1000
  num_time_steps = 13

  inference_func = util_experiments.wrap_dummy_inference(num_users)

  z_states = inference_func(None, None, None, num_time_steps)

  np.testing.assert_array_almost_equal(
    z_states.shape, [num_users, num_time_steps, 4])
