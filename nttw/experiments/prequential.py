"""Experiments related to sequential predicton and simulation."""
import datetime
import json
import numpy as np
from nttw import constants, util
import os
import random
from typing import Any, Dict, Iterable, List, Tuple, Union


def dump_results(
    datadir: str, **kwargs):
  fname = os.path.join(datadir, "prec_recall_ir.npz")
  with open(fname, 'wb') as fp:
    np.savez(fp, **kwargs)


def dump_results_json(
    datadir: str,
    cfg: Dict[str, Any],
    **kwargs):
  """Dumps the results of an experiment to JSONlines."""

  kwargs["time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  kwargs["slurm_id"] = str(os.getenv('SLURM_JOB_ID'))  # Defaults to 'none'
  kwargs["slurm_name"] = str(os.getenv('SLURM_JOB_NAME'))  # Defaults to 'none'
  kwargs["sweep_id"] = str(os.getenv('SWEEPID'))  # Defaults to 'none'

  model_keys = [
    "p0", "p1", "alpha", "beta", "prob_g", "prob_h", "sib_mult", "damping",
    "noisy_test", "num_days_window", "num_buckets", "quantization",
    "threshold_quarantine", "freeze_backwards", "num_rounds"]
  data_keys = [
    "num_users", "num_time_steps", "fraction_quarantine", "num_days_quarantine",
    "fraction_test", "do_conditional_testing", "fraction_stale"]

  for key in model_keys:
    kwargs[f"model.{key}"] = cfg["model"][key]

  for key in data_keys:
    kwargs[f"data.{key}"] = cfg["data"][key]

  fname = os.path.join(datadir, "results.jl")
  with open(fname, 'a') as fp:
    fp.write(json.dumps(kwargs) + "\n\r")


def init_states_observations(
    num_users: int, num_days: int
    ) -> Tuple[np.ndarray, List[constants.Observation]]:
  """Initializes the states and observations for prequential simulation."""
  states = np.zeros((num_users, num_days), dtype=np.int16)

  num_users_E_at_start = 2
  observations_all = []

  for user in range(num_users_E_at_start):
    states[user, 0] = 1
    observations_all.append((user, 2, 1))
  return states, observations_all


def simulate_one_day(
    states: np.ndarray, contacts_list: List[constants.Contact], timestep: int,
    p0: float, p1: float, g: float, h: float) -> np.ndarray:
  """Simulates the states for one day, given the contacts."""
  # Sample states at t='timestep' given states up to and including 'timestep-1'

  num_users = states.shape[0]

  # Construct counter on every call, because contacts may change
  infect_counter = util.InfectiousContactCount(
    contacts=contacts_list,
    samples=None,
    num_users=num_users,
    num_time_steps=timestep+1,
  )
  for user in range(num_users):
    if states[user][timestep-1] == 0:
      log_f_term = np.log(1-p0)
      for _, user_u, _ in infect_counter.get_past_contacts_at_time(
          user, timestep-1):
        if states[user_u][timestep-1] == 2:
          log_f_term += np.log(1-p1)
      p_state_up = 1-np.exp(log_f_term)
    elif states[user][timestep-1] == 1:
      p_state_up = g
    elif states[user][timestep-1] == 2:
      p_state_up = h
    elif states[user][timestep-1] == 3:
      p_state_up = 0

    # Increase state according to random sample
    state_up = (random.random() < p_state_up)
    states[user][timestep] = states[user][timestep-1] + state_up
  return states


def get_observations_one_day(
    states: np.ndarray, users_to_observe: List[int], timestep: int,
    p_obs_infected: List[float], p_obs_not_infected: List[float]):
  """Makes observations for tests on one day."""

  assert len(states.shape) == 1

  np.testing.assert_allclose(p_obs_infected[0] + p_obs_infected[1], 1.)
  np.testing.assert_allclose(p_obs_not_infected[0] + p_obs_not_infected[1], 1.)

  for user in users_to_observe:
    state_user = states[user]
    sample_prob = p_obs_infected if state_user == 2 else p_obs_not_infected
    outcome = np.random.choice(2, p=sample_prob)
    yield (int(user), int(timestep), int(outcome))


def calc_prec_recall(
    states: np.ndarray, users_to_quarantine: np.ndarray) -> Tuple[float, float]:
  """Calculates precision and recall for quarantine assignments."""
  assert len(states.shape) == 1

  users_quarantine_array = np.zeros_like(states)
  users_quarantine_array[users_to_quarantine] = 1.
  states_e_i = np.logical_or(
    states == 1,
    states == 2,
  )

  true_positives = np.sum(np.logical_and(states_e_i, users_quarantine_array))

  precision = true_positives / np.sum(users_quarantine_array)
  recall = true_positives / np.sum(states_e_i)
  return precision, recall


def select_quarantine_users(
    traces: np.ndarray, threshold: float = .3) -> np.ndarray:
  """Selects users to quarantine if p(z in {E,I}) exceeds threshold."""
  return np.where((traces[:, -1, 1] + traces[:, -1, 2]) > threshold)[0]


def select_quarantine_users_max(
    traces: np.ndarray, num_quarantine=20) -> np.ndarray:
  """Selects users to quarantine based on highest p(z in {E,I}) score."""
  score = traces[:, -1, 1] + traces[:, -1, 2]
  return np.argsort(score)[-num_quarantine:]


def remove_quarantine_users(
    contacts: List[constants.Contact],
    quarantine_users: Union[List[int], np.ndarray],
    t_start: int,
    t_delta: int) -> List[constants.Contact]:
  """Removes quarantined users from contact list.

  A contact will be removed if EITHER the user u or user v is being quarantined
  and the timestep is GREATER OR EQUAL t_start and LESS THAN t_start+t_delta.
  """
  def filter_func(contact):
    return not (
      ((contact[0] in quarantine_users) or (contact[1] in quarantine_users))
      and (contact[2] >= t_start)
      and (contact[2] < (t_start+t_delta)))
  return list(filter(filter_func, contacts))


def get_evidence_obs(
    observations: List[constants.Observation],
    z_states: np.ndarray,
    alpha: float,
    beta: float) -> float:
  """Calculates evidence for the observations, integrating out the states."""
  p_obs_infected = [alpha, 1-alpha]
  p_obs_not_infected = [1-beta, beta]

  log_like = 0.
  for obs in observations:
    user, timestep, outcome = obs[0], obs[1], obs[2]
    p_inf = z_states[user, timestep, 2]

    log_like += np.log(
      p_inf*p_obs_infected[outcome] + (1-p_inf)*p_obs_not_infected[outcome]
      + 1E-9)
  return log_like


def decide_tests(
    scores_infect: np.ndarray,
    test_include: np.ndarray,
    num_tests: int) -> Tuple[np.ndarray, np.ndarray]:
  assert num_tests < len(scores_infect)
  assert scores_infect.shape == test_include.shape

  users_to_test = np.argsort(scores_infect*test_include)[-num_tests:]
  return users_to_test


def remove_positive_users(observations, test_include: np.ndarray) -> np.ndarray:
  # o[2] is the outcome of the test: 1 for positive test, 0 for negative test
  for obs in filter(lambda o: o[2] > 0, observations):
    test_include[obs[0]] = 0.
  return test_include


def delay_contacts(
    contacts: List[constants.Contact], delay: int
    ) -> Iterable[constants.Contact]:
  """Offsets the contacts or observations by a number of days."""

  for contact in contacts:
    yield (contact[0], contact[1], contact[2] + delay, contact[3])


def offset_edges(edges: List[Any], offset: int) -> Iterable[Any]:
  """Offsets the contacts or observations by a number of days."""

  if (offset == 0) or (len(edges) == 0):
    yield from edges
    return

  if len(edges[0]) == 3:  # Observation
    for edge in edges:
      if edge[1] >= offset:
        yield (edge[0], edge[1] - offset, edge[2])

  elif len(edges[0]) == 4:  # Contact
    for edge in edges:
      if edge[2] >= offset:
        yield (edge[0], edge[1], edge[2] - offset, edge[3])

  else:
    raise ValueError("Edges must be tuples of 3 or 4")
