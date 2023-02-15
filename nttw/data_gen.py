"""Generate data for CRISP-like graph structures and experiments."""

import random
from typing import Union
import warnings

import numpy as np


def init_contacts(
    num_users: int,
    num_time_steps: int,
    qIbar: float = 20.0,
    R0: Union[float, np.ndarray] = 2.5,
    p1: float = 0.01,
    R0_mit=(2.5, 0.5),
    t_mit=None,
    H=None,
    seed: int = 42):
  """Generate random contacts according to parameters."""

  random.seed(seed)
  np.random.seed(seed+1)

  if isinstance(R0, float):
    print(f"Ave contacts per user per day {R0 / qIbar / p1:.2f}")
    R0 = np.ones(num_time_steps) * R0
  elif isinstance(R0, np.ndarray):
    assert len(R0) == num_time_steps
  else:
    raise ValueError(f"parameter R0 must be float of np.array, was {type(R0)}")

  # Precompute all contacts in the constructor
  user_list = np.arange(num_users)
  user_list0 = user_list[:, np.newaxis]
  user_list1 = user_list[np.newaxis, :]

  # a lower triangular binary mask of potention (unidirectional) contacts
  mask = (user_list[:, np.newaxis] > user_list[np.newaxis, :])
  potential_contacts = list(zip(*np.where(mask)))

  # if H is set, we have H local cliques
  if H is not None:
    # the mask for in-clique contatcs
    maskb = mask * (user_list0 - user_list1 < user_list0 % H + 1)
    # the mask for inter-clique contacts
    maska = mask * (~maskb)
    pa = R0_mit[1] / qIbar / p1 / (num_users - H)
    pb = R0_mit[0] / qIbar / p1 / (H - 1)
    if pb > 1.0:
      warnings.warn(("Mitigation results in decreased nominal R0, "
                     "increase H to suppress this warning!"))

    potential_contactsa = list(zip(*np.where(maska)))
    potential_contactsb = list(zip(*np.where(maskb)))

  def sample(potential_contacts, p0, time_step):
    N = len(potential_contacts)
    num_sample = np.random.binomial(N, p0)
    if num_sample == 0:
      print(f"Warning: empty contact list at timestep {time_step}")
      return np.array([])
    c = np.array(random.sample(potential_contacts, num_sample))

    # Concatenate contacts with features
    c = np.c_[c, np.full_like(c[:, 0], time_step), np.ones_like(c[:, 0])]

    # Include all symmetric contacts
    return np.r_[c, c[:, [1, 0, 2, 3]]]

  for t in range(num_time_steps):

    if t_mit is None or t < t_mit:
      # if p0 > 1, then there's not enough users to obtain the specified R0
      p0 = R0[t] / qIbar / p1 / (num_users - 1)
      yield sample(potential_contacts, p0, t)
    else:
      yield np.r_[
        sample(potential_contactsa, pa, t),
        sample(potential_contactsb, pb, t)]
