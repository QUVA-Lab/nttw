"""Utility functions to run Inference with SIB

Retrieved on Sept 22, 2022 from
  'github.com/sibyl-team/epidemic_mitigation/blob/master/src/abm_utils.py'

"""
import numpy as np
from scipy import stats
import sib
from typing import Tuple


def gamma_params(mean: float, stddev: float) -> Tuple[float, float]:
  scale = (stddev**2)/mean
  return (mean/scale, scale)


def gamma_pdf_array(num_time_steps: int, mu: float, sigma: float) -> np.ndarray:
  """
  discrete gamma function:
  T: len(array) = T+1
  mu: mu of gamma
  sigma: std of gammas
  """
  k, scale = gamma_params(mu, sigma)
  gamma_array = stats.gamma.pdf(range(num_time_steps + 1), k, scale=scale)
  return gamma_array


def construct_dynamics(num_time_steps: int):
  """Constructs dynamics distributions according to SIB code.

  Follows 'github.com/sibyl-team/epidemic_mitigation/blob/master/...
           epi_mitigation.ipynb', retrieved on Sept 22, 2022
  """
  prob_i = sib.PiecewiseLinear(
    sib.RealParams(list(0.25 * gamma_pdf_array(num_time_steps+1, 6, 2.5))))
  prob_r = sib.PiecewiseLinear(
    sib.RealParams(list(stats.gamma.sf(
      range(num_time_steps+1), 10., scale=1.7452974337097158))))
  return prob_i, prob_r


def make_sib_params(num_time_steps: int, p0: float, recovery_days: float):
  """Makes the parameters to init inference with SIB.

  As of Sept 23, 2022, SIB has two implementations online (confusing...)
  When recovery_days is positive, implementation [1] is used. When recovery_days
  is negative, implementation [2] is used.

  [1] github.com/sibyl-team/sib/blob/master/examples/dummy_test.ipynb
  [2] github.com/sibyl-team/epidemic_mitigation/blob/master/epi_mitigation.ipynb
  """
  if recovery_days > 0:
    # This is how one of the notebooks for
    return sib.Params(
      prob_r=sib.Exponential(mu=1/recovery_days),
      pseed=p0,  # Prior to be infected at time zero
      pautoinf=p0)

  # Make Parameters
  prob_i, prob_r = construct_dynamics(num_time_steps=num_time_steps)
  return sib.Params(
    prob_i=prob_i, prob_r=prob_r, pseed=p0, psus=0.55, pautoinf=p0)
