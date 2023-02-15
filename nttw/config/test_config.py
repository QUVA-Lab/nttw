"""Tests for config.py."""

from nttw.config import config
import numpy as np


def test_clean_hierarchy():
  data = {"a": 1, "b": 2}
  result = config.clean_hierarchy(data)
  assert result == data

  with np.testing.assert_raises(AssertionError):
    data = {"model.a": 1, "b": 2, "model": {"c": 3}, "data": {"d": 4}}
    result = config.clean_hierarchy(data)

  data = {
    "model.a": 1,
    "data.d": 1,
    "b": 2,
    "model": {"a": -1., "c": 3},
    "data": {"d": 4}}
  result = config.clean_hierarchy(data)
  expected = {"b": 2, "model": {"a": 1, "c": 3}, "data": {"d": 1}}
  assert result == expected
