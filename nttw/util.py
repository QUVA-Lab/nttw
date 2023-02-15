"""Utility functions for inference in CRISP-like models."""

import functools
import itertools
from time import time  # pylint: disable=unused-import
import jax.numpy as jnp
import math
from nttw import constants, logger
from numba import njit
import numpy as np
import os
import socket
from typing import (
  Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union)

Array = Union[np.ndarray, jnp.array]


class InfectiousContactCount:
  """Counter for infectious contacts."""

  def __init__(self,
               contacts: List[constants.Contact],
               samples: Optional[Mapping[int, Union[Array, List[int]]]],
               num_users: int,
               num_time_steps: int):
    self._counts = np.zeros((num_users, num_time_steps + 1), dtype=np.int32)
    self._num_time_steps = num_time_steps

    # TODO: WARNING: This constructor assumes that the contacts don't change!
    self.past_contacts = {
      user: [[] for _ in range(num_time_steps)] for user in range(num_users)}
    self.future_contacts = {
      user: [[] for _ in range(num_time_steps)] for user in range(num_users)}

    for contact in contacts:
      user_u = contact[0]
      user_v = contact[1]
      timestep = contact[2]
      if timestep < num_time_steps:
        self.past_contacts[user_v][timestep].append(
          (timestep, user_u, contact[3]))
        self.future_contacts[user_u][timestep].append(
          (timestep, user_v, contact[3]))

      if samples:
        trace_u = samples[user_u]
        if state_at_time_cache(*trace_u, contact[2]) == 2:
          self._counts[user_v][contact[2]+1] += 1

  def get_past_contacts_slice(
      self, user_slice: Union[List[int], np.ndarray]) -> np.ndarray:
    """Outpets past contacts as a NumPy array, for easy pickling.

    Defaults to -1 when no past contacts exist (and to fill sparse array)
    """
    past_contacts = []
    max_messages = -1  # Output ndarray will be size of longest list

    for user in user_slice:
      pc_it = itertools.chain.from_iterable(self.past_contacts[user])
      pc_array = np.array(
        list(map(lambda x: [x[0], x[1], x[2][0]], pc_it)), dtype=np.int16)
      past_contacts.append(pc_array)

      # Update longest amount of messages
      max_messages = max((max_messages, len(pc_array)))

    # Default to -1 for undefined past contacts
    pc_tensor = -1 * np.ones(
      (len(user_slice), max_messages+1, 3), dtype=np.int16)
    for i, user in enumerate(user_slice):
      num_contacts = len(past_contacts[i])
      if num_contacts > 0:
        pc_tensor[i][:num_contacts] = past_contacts[i]

    return pc_tensor

  def num_inf_contacts(self, user: int, time_step: int):
    return self._counts[user, time_step]

  def update_infect_count(
      self, user: int, trace: Union[List[int], np.ndarray],
      remove: bool = False):
    t0, de, di = trace
    for timestep in range(t0+de, t0+de+di):
      for (_, user_v, feature) in self.future_contacts[user][timestep]:
        assert int(feature[0]) == 1, "Only implemented for feature_val==1"
        add = -1 if remove else 1
        self._counts[user_v][timestep+1] += add

  def get_future_contacts(self, user: int):
    yield from itertools.chain.from_iterable(self.future_contacts[user])

  def get_past_contacts(self, user: int):
    yield from itertools.chain.from_iterable(self.past_contacts[user])

  def get_past_contacts_at_time(self, user: int, timestep: int):
    yield from self.past_contacts[user][timestep]


def state_at_time(days_array, timestamp):
  """Calculates the SEIR state at timestamp given the Markov state.

  Note that this function is slower than 'state_at_time_cache' when evaluating
  only one data point.
  """
  if isinstance(days_array, list):
    days_array = np.array(days_array, ndmin=2)
  elif len(days_array.shape) == 1:
    days_array = np.expand_dims(days_array, axis=0)

  days_cumsum = np.cumsum(days_array, axis=1)
  days_binary = (days_cumsum <= timestamp).astype(np.int)

  # Append vector of 1's such that argmax defaults to 3
  days_binary = np.concatenate(
    (days_binary, np.zeros((len(days_binary), 1))), axis=1)
  state = np.argmin(days_binary, axis=1)
  return state


# @functools.lru_cache()  # Using cache gives no observable speed up for now
def state_at_time_cache(t0: int, de: int, di: int, t: int) -> int:
  if t < t0:
    return 0
  if t < t0+de:
    return 1
  if t < t0+de+di:
    return 2
  return 3


def gather_infected_precontacts(
    num_time_steps: int,
    samples_current: Mapping[int, Union[List[int], np.ndarray]],
    past_contacts: Iterable[Tuple[int, int, List[int]]]):
  """Gathers infected precontacts.

  For all past contacts, check if the contact was infected according to samples.
  """

  num_infected_preparents = np.zeros((num_time_steps))

  for (t_contact, user_u, features) in past_contacts:
    assert len(features) == int(features[0]) == 1, (
      "Code only implemented for singleton feature at 1")
    trace_u = samples_current[user_u]
    state_u = state_at_time_cache(*trace_u, t_contact)
    if state_u == 2:
      num_infected_preparents[t_contact] += features[0]

  return num_infected_preparents


def calc_grad_p1(
    num_infected_preparents: np.ndarray,
    probab_0: float,
    probab_1: float,
    time_s: int):
  """Calculate the gradient with respect to transmission parameter, p1."""

  # Number of infectious preparents at final timestep in state S
  N_term = num_infected_preparents[time_s-1]

  # Gradient for increasing p1, according to activity at final timestep in S
  condition = (1-probab_0) / (1-(1-probab_0)*(1-probab_1)**N_term)
  gradient_up = condition * N_term * (1-probab_1)**(N_term-1)
  assert gradient_up >= 0

  # Gradient down, according to activity prior to final timestep in S
  gradient_down = 1 / (1-probab_1) * np.sum(num_infected_preparents[:time_s-1])
  assert gradient_down >= 0
  return gradient_up - gradient_down


def precompute_p_u_t(
    time_total: int,
    user_u: int,
    infect_counter: InfectiousContactCount,
    samples_current: Dict[int, Array],
    probab_channels: np.ndarray) -> np.ndarray:
  """Precompute the p_{u,t} array.

  Note that the array is 0-indexed. So time t=1 will be found at index p_u_t[1].
  The zero-th element defaults to value=1 when not used.
  """

  log_p_u_t = np.zeros((time_total + 1))
  for (t_contact, user_other, features) in infect_counter.get_past_contacts(
      user_u):
    # Implement some caching here!
    state_other = state_at_time_cache(*samples_current[user_other],
                                      t_contact)
    if int(state_other) == 2:  # Being infected
      noisy_or = np.prod((1 - probab_channels)**np.array(features))
      if t_contact <= time_total:
        log_p_u_t[t_contact] += np.log(noisy_or)
  return log_p_u_t


def precompute_b_z(
  total_time: int,
  user_u: int,
  infect_counter: InfectiousContactCount,
  samples_current: Dict[int, Array],
  probab_channels: np.ndarray,
  probab_0: float,
):
  """Precompute the Bz terms for Gibbs sampling.

  Notation follows the original CRISP paper.
  """
  assert len(probab_channels) == 1, "only implemented for one channel now"
  p1 = probab_channels[0]
  # Precompute b_z_t_i and b_z_t_noti
  log_b_z_ratio = np.zeros((total_time + 1))

  for (t_contact, user_v, _) in infect_counter.get_future_contacts(user_u):
    # TODO alter function such that t and t+1 computed together
    state_other = state_at_time_cache(*samples_current[user_v], t_contact)
    if state_other == 0:  # Other is susceptible
      state_other_next = state_at_time_cache(*samples_current[user_v],
                                             t_contact + 1)

      # TODO implement for multiple channels
      # TODO implement for when feature_value != 1
      parent_contacts = infect_counter.num_inf_contacts(user_v, t_contact+1)
      feature_value = 1
      if state_other_next == 0:
        # if contact s->s, then ratio is only 1-p_j
        log_b_z_ratio[t_contact] += np.log(feature_value * (1. - p1))
      elif state_other_next == 1:
        # if contact s->e, then calculate ratio
        b_incl_i = 1.-(1.-probab_0)*pow(1.-p1, parent_contacts+feature_value)
        b_not_i = 1.-(1.-probab_0)*pow(1.-p1, parent_contacts)
        log_b_z_ratio[t_contact] += np.log(b_incl_i) - np.log(b_not_i)
  return log_b_z_ratio


def calc_c_z_u(
    potential_sequences: np.ndarray,
    observations: List[constants.Observation],
    num_users: int,
    alpha: float,
    beta: float):
  """Precompute the Cz terms.

  Notation follows the original CRISP paper.
  """

  # Map outcome to PMF over health state
  probabs = {0: [alpha, 1-beta],
             1: [1-alpha, beta]}

  log_prob_obs = {user: np.zeros((len(potential_sequences)))
                  for user in range(num_users)}

  seq_cumsum = np.cumsum(potential_sequences, axis=1)

  # This implementation assumes sparse observations. With dense observations,
  # more efficient to calculate seq_binary outside for-loop
  for obs in observations:
    probab_tuple = probabs[obs[2]]
    user_u = obs[0]

    # Append ones to default to state R
    seq_binary = (seq_cumsum > obs[1]).astype(np.int)
    seq_binary = np.concatenate((seq_binary, np.ones((len(seq_binary), 1))),
                                axis=1)
    state = np.argmax(seq_binary, axis=1)
    log_prob_obs[user_u] += np.log(
      np.where(state == 2, probab_tuple[0], probab_tuple[1]))

  return log_prob_obs


def calc_log_a_start(
    seq_array: np.ndarray,
    probab_0: float,
    g: float,
    h: float) -> np.ndarray:
  """Calculate the basic A terms.

  This assumes no contacts happen. Thus the A terms are simple Geometric
  distributions. When contacts do occur, subsequent code would additional log
  terms.
  """
  if isinstance(seq_array, list):
    seq_array = np.stack(seq_array, axis=0)

  num_sequences = seq_array.shape[0]
  log_A_start = np.zeros((num_sequences))

  time_total = np.max(np.sum(seq_array, axis=1))

  # Due to time in S
  # Equation 17
  term_t = (seq_array[:, 0] >= time_total).astype(np.float32)
  log_A_start += (seq_array[:, 0]-1) * np.log(1-probab_0)

  # Due to time in E
  term_e = ((seq_array[:, 0] + seq_array[:, 1]) >= time_total).astype(np.int64)
  log_A_start += (1-term_t) * (
    (seq_array[:, 1]-1)*np.log(1-g) + (1-term_e)*np.log(g)
  )

  # Due to time in I
  term_i = (seq_array[:, 0] + seq_array[:, 1] + seq_array[:, 2]) >= time_total
  log_A_start += (1-term_e) * (
    (seq_array[:, 2]-1)*np.log(1-h) + (1-term_i.astype(np.int64))*np.log(h))
  return log_A_start


def sample_q(
    param: Array,
    num_samples: int,
    num_time: int,
    collapse: bool = True) -> Array:
  """Sample trajectories from the variational distribution q."""
  # TODO: update this function for untied unrolled variational q!
  # Assumes param[0] is the probability of starting in state 0
  # Assumes param[i] for i in range(1,4) are the probs of staying in state i
  transition = np.concatenate((1.-param[1:4], [0.]))  # Probability of moving up

  samples = [(np.random.rand(num_samples) > param[0]).astype(np.int32)]
  for i in range(1, num_time):
    p_move_up = np.take(transition, samples[i-1], axis=0)

    move_up = (np.random.rand(num_samples) < p_move_up).astype(np.int32)

    samples.append(samples[i-1] + move_up)

  samples = np.stack(samples, axis=1)
  if collapse:
    t0 = np.sum(samples == 0, axis=1)
    t1 = np.sum(samples == 1, axis=1)
    t2 = np.sum(samples == 2, axis=1)

    samples = np.stack([t0, t1, t2], axis=1)

  return samples


def q_log_tail_prob(pdf_values):
  """Calculates the log probs and log tail probs.

  pdf_values are assumed 0-indexed. So any array would start with 0 and then
  actual values starting day 1. For a Geometric(.5), this would be
  [0, .5, .5**2, .5**3, ...]

  """
  logpdf = np.log(pdf_values)
  logpdf_tail = np.concatenate(
    ([0.], np.log(1 - np.cumsum(pdf_values) + 1E-12)[:-1]))
  return logpdf, logpdf_tail


def iter_state(ts, te, ti, tt):
  yield from itertools.repeat(0, ts)
  yield from itertools.repeat(1, te)
  yield from itertools.repeat(2, ti)
  yield from itertools.repeat(3, tt-ts-te-ti)


def calc_expected_days_rolled(
    variational_params: Array, num_time_steps: int) -> np.ndarray:
  """Calculates the expected days in the I state, given variational params.

  Args:
    variational_params: Array of variational parameters. Assumes the same
    structure as for inference.py, so the 0th param is \\alpha, then every
    num_time_steps parameters correspond to \\beta_0, \\beta_1 and \\beta_2.

  Returns:
    Scalar float, being the expected days in state I.
  """
  assert len(variational_params) >= (3*num_time_steps+1)
  beta2_vec = variational_params[2*num_time_steps+1:3*num_time_steps+1]

  p_stay_extend = np.concatenate(([1], beta2_vec))
  cumprod = np.exp(np.cumsum(np.log(p_stay_extend)))
  return np.sum(cumprod[:-1] * (1.-beta2_vec) * np.arange(1, num_time_steps+1))


def calc_expected_days(
    variational_params: Dict[int, Array], seq_array: Array) -> np.ndarray:
  """Calculates the expected days in the I state, given variational params.

  Args:
    variational_params: Dictionary of variational paramters. Assumes the same
    structure as for inference.py, so the 0th param is \\alpha, then every
    num_time_steps parameters correspond to \\beta_0, \\beta_1 and \\beta_2.
    seq_array: array of size [num_sequences, 3] with all possible sequences

  Returns:
    Scalar float, being the expected days in state I.
  """
  num_time_steps = np.max(seq_array)
  assert len(variational_params[0]) >= (3*num_time_steps+1)

  num_users = len(variational_params)
  t_I_array = np.zeros((num_users))

  for user in range(num_users):
    log_q = enumerate_log_q_values_untied_unrolled_beta_jax(
      variational_params[user], seq_array, time_total=num_time_steps)

    q_all = np.exp(np.array(log_q))
    np.testing.assert_almost_equal(np.sum(q_all), 1.0, decimal=4)

    t_I_array[user] = np.inner(seq_array[:, 2], q_all)
  return np.mean(t_I_array)


def state_seq_to_time_seq(
    state_seqs: Union[Array, List[List[int]]],
    time_total: int) -> Array:
  """Unfolds trace tuples to full traces of SEIR.

  Args:
    state_seqs: Array in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    Array of [num_sequences, time_total], with values in {0,1,2,3}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))
  return np.array(list(iter_time_seq))


def state_seq_to_hot_time_seq(
    state_seqs: Union[Array, List[List[int]]],
    time_total: int) -> Array:
  """Unfolds trace tuples to one-hot traces of SEIR.

  Args:
    state_seqs: Array in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    Array of [num_sequences, time_total, 4], with values in {0,1}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))

  states = np.zeros((len(state_seqs), time_total, 4))
  for i, time_seq in enumerate(iter_time_seq):
    states[i] = np.take(np.eye(4), np.array(time_seq), axis=0)
  return states


def iter_prior(
    lqs: Union[List[float], np.ndarray],
    lqs_tail: Union[List[float], np.ndarray],
    lqe: Union[List[float], np.ndarray],
    lqe_tail: Union[List[float], np.ndarray],
    lqi: Union[List[float], np.ndarray],
    lqi_tail: Union[List[float], np.ndarray],
    time_total: int) -> Iterator[Tuple[Tuple[int, int, int], float]]:
  """Iterate over all sequences and yield trace and prior probability."""
  p_total = 0
  for t0 in range(time_total+1):
    if t0 == time_total:
      logp = lqs_tail[t0]
      yield (t0, 0, 0), logp
      p_total += np.exp(logp)
    else:
      for de in range(1, time_total-t0+1):
        if t0+de == time_total:
          logp = lqs[t0] + lqe_tail[de]
          yield (t0, de, 0), logp
          p_total += np.exp(logp)
        else:
          for di in range(1, time_total-t0-de+1):
            if t0+de+di == time_total:
              logp = lqs[t0] + lqe[de] + lqi_tail[di]
              yield (t0, de, di), logp
              p_total += np.exp(logp)
            else:
              logp = lqs[t0] + lqe[de] + lqi[di]
              yield (t0, de, di), logp
              p_total += np.exp(logp)
  assert (p_total-1.) < 1E-5, f'Prior does not sum to 1, but is {p_total}'


def iter_sequences(time_total: int, start_se=True):
  """Iterate possible sequences.

  Assumes that first time step can be either S or E.
  """
  for t0 in range(time_total+1):
    if t0 == time_total:
      yield (t0, 0, 0)
    else:
      e_start = 1 if (t0 > 0 or start_se) else 0
      for de in range(e_start, time_total-t0+1):
        if t0+de == time_total:
          yield (t0, de, 0)
        else:
          i_start = 1 if (t0 > 0 or de > 0 or start_se) else 0
          for di in range(i_start, time_total-t0-de+1):
            if t0+de+di == time_total:
              yield (t0, de, di)
            else:
              yield (t0, de, di)


def generate_sequence_days(time_total: int):
  """Iterate possible sequences.

  Assumes that first time step must be S.
  """
  # t0 ranges in {T,T-1,...,1}
  for t0 in range(time_total, 0, -1):
    # de ranges in {T-t0,T-t0-1,...,1}
    # de can only be 0 when time_total was already spent
    de_start = min((time_total-t0, 1))
    non_t0 = time_total - t0
    for de in range(de_start, non_t0+1):
      # di ranges in {T-t0-de,T-t0-de-1,...,1}
      # di can only be 0 when time_total was already spent
      di_start = min((time_total-t0-de, 1))
      non_t0_de = time_total - t0 - de
      for di in range(di_start, non_t0_de+1):
        yield (t0, de, di)


def enumerate_log_q_values_normalised(
    params: Array,
    sequences: Array) -> Array:
  """Enumerate values of log_q for variational parameters."""
  b1, b2 = params[1], params[2]
  num_time_steps = np.max(sequences)

  Z1 = (1-jnp.power(b1, num_time_steps))
  Z2 = (1-jnp.power(b2, num_time_steps))
  lq1_tail = jnp.log(
    jnp.array([1-(1-b1**i)/Z1 for i in range(1, num_time_steps+1)]))
  lq1_tail = jnp.concatenate((jnp.array([0, 0]), lq1_tail))
  lq2_tail = jnp.log(
    jnp.array([1-(1-b2**i)/Z2 for i in range(1, num_time_steps+1)]))
  lq2_tail = jnp.concatenate((jnp.array([0, 0]), lq2_tail))

  lq2 = jnp.log(
    jnp.array([(1-b2)*b2**(i-1)/Z2 for i in range(1, num_time_steps+1)]))
  lq2 = jnp.concatenate((jnp.array([0]), lq2))

  log_q_z = jnp.zeros((len(sequences)))

  # Terms due to state S
  term_s = (sequences[:, 0] >= num_time_steps).astype(np.float32)
  log_q_z += (sequences[:, 0] * jnp.log(params[0])
              + (1-term_s)*jnp.log(1-params[0]))

  # Terms due to state E
  lq1_take = jnp.log(1-b1) + (sequences[:, 1]-1) * jnp.log(b1) - jnp.log(Z1)
  term_e = (sequences[:, 0]+sequences[:, 1] >= num_time_steps
            ).astype(np.float32)
  log_q_z += (1-term_s) * jnp.where(
    term_e,
    jnp.take(lq1_tail, sequences[:, 1], axis=0),
    lq1_take)

  # Terms due to state I
  term_i = (sequences[:, 0]+sequences[:, 1]+sequences[:, 2] >= num_time_steps
            ).astype(np.float32)
  log_q_z += (1-term_e) * jnp.where(
    term_i,
    jnp.take(lq2_tail, sequences[:, 2], axis=0),
    jnp.take(lq2, sequences[:, 2], axis=0))

  return log_q_z


def enumerate_log_prior_values(
    params_start: Array,
    params: Array,
    sequences: Array,
    time_total: int) -> Array:
  """Enumerate values of log prior."""
  np.testing.assert_almost_equal(np.sum(params_start), 1.)

  b0, b1, b2 = params[0], params[1], params[2]

  start_s = reach_s = sequences[:, 0] > 0
  start_e = (1-start_s) * (sequences[:, 1] > 0)
  start_i = (1-start_s) * (1-start_e) * (sequences[:, 2] > 0)
  start_r = (1-start_s) * (1-start_e) * (1-start_i)

  reach_e = (sequences[:, 1] > 0)
  reach_i = (sequences[:, 2] > 0)
  reach_r = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) < time_total

  log_q_z = np.zeros((len(sequences)))

  # Terms due to start state
  log_q_z += start_s * np.log(params_start[0] + 1E-12)
  log_q_z += start_e * np.log(params_start[1] + 1E-12)
  log_q_z += start_i * np.log(params_start[2] + 1E-12)
  log_q_z += start_r * np.log(params_start[3] + 1E-12)

  # Terms due to days spent in S
  log_q_z += np.maximum(sequences[:, 0]-1, 0.) * np.log(b0)
  log_q_z += reach_s * reach_e * np.log(1-b0)  # Only when transit to E is made

  # Terms due to days spent in E
  log_q_z += np.maximum(sequences[:, 1] - 1, 0.) * np.log(b1)
  log_q_z += reach_e * reach_i * np.log(1-b1)  # Only when transit to I is made

  # Terms due to days spent in I
  log_q_z += np.maximum(sequences[:, 2] - 1, 0.) * np.log(b2)
  log_q_z += reach_i * reach_r * np.log(1-b2)  # Only when transit to R is made

  return log_q_z


def enumerate_start_belief(
    seq_array: np.ndarray, start_belief: np.ndarray) -> np.ndarray:
  """Calculates the start_belief for all enumerated sequences."""
  assert seq_array.shape[1] == 3
  assert start_belief.shape == (4,)

  start_s = seq_array[:, 0] > 0
  start_e = (1.-start_s) * (seq_array[:, 1] > 0)
  start_i = (1.-start_s) * (1.-start_e) * (seq_array[:, 2] > 0)
  start_r = np.sum(seq_array, axis=1) == 0

  return (
    start_belief[0] * start_s
    + start_belief[1] * start_e
    + start_belief[2] * start_i
    + start_belief[3] * start_r)


def enumerate_log_q_values_jax(
    params: Array,
    sequences: Array,
    time_total: int) -> Array:
  """Enumerate values of log_q for variational parameters."""
  a0, b0, b1, b2 = params[0], params[1], params[2], params[3]

  log_q_z = jnp.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * jnp.log(a0) + (1-start_s) * jnp.log(1-a0)
  log_q_z += start_s * (sequences[:, 0]-1)*jnp.log(b0)
  log_q_z += start_s * (1-term_s) * jnp.log(1-b0)

  # Terms due to e
  log_q_z += (1-term_s) * (sequences[:, 1] - 1) * jnp.log(b1)
  log_q_z += reach_i * jnp.log(1-b1)

  # Terms due to I
  log_q_z += reach_i * (sequences[:, 2]-1) * jnp.log(b2)
  log_q_z += reach_i * (1-term_i) * jnp.log(1-b2)

  return log_q_z


def enumerate_log_q_values_untied_beta_jax(
    params: Array,
    sequences: Array,
    time_total: int) -> Array:
  """Enumerate values of log_q for untied variational parameters."""
  assert(len(params.shape)) == 1, "Expect an array of [num_time, 4]"
  assert len(params) == (3*time_total + 1)
  # Interpret parameters as: a0, b01, b02, ..., b0T, b1, b2
  a0 = params[0]
  b0 = params[1:time_total+1]
  b1 = params[time_total+1:2*time_total+1]
  b2 = params[2*time_total+1:3*time_total+1]

  b0_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b0))))
  b1_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b1))))
  b2_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b2))))

  log_q_z = jnp.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * jnp.log(a0) + (1-start_s) * jnp.log(1-a0)
  log_q_z += start_s * jnp.take(b0_cumsum, sequences[:, 0]-1)
  log_q_z += start_s * (1-term_s) * jnp.log(1-jnp.take(b0, sequences[:, 0]-1))

  # Terms due to e
  log_q_z += (1-term_s) * jnp.take(b1_cumsum, sequences[:, 1]-1)
  log_q_z += reach_i * jnp.log(1-jnp.take(b1, sequences[:, 1]-1))

  # Terms due to I
  log_q_z += reach_i * jnp.take(b2_cumsum, sequences[:, 2]-1)
  log_q_z += reach_i * (1-term_i) * jnp.log(1-jnp.take(b2, sequences[:, 2]-1))

  return log_q_z


def enumerate_log_q_values_untied_unrolled_beta_jax(
    params: Array,
    sequences: Array,
    time_total: int) -> Array:
  """Enumerate values of log_q for untied and unrolled variational params."""
  assert(len(params.shape)) == 1, "Expect an array of [num_time, 4]"
  assert len(params) == (3*time_total + 1)
  # Interpret parameters as: a0, b01, b02, ..., b0T, b1, b2
  a0 = params[0]
  b0 = params[1:time_total+1]
  b1 = params[time_total+1:2*time_total+1]
  b2 = params[2*time_total+1:3*time_total+1]

  b0_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b0))))
  b1_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b1))))
  b2_cumsum = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.log(b2))))

  log_q_z = jnp.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * jnp.log(a0) + (1-start_s) * jnp.log(1-a0)
  log_q_z += start_s * jnp.take(b0_cumsum, sequences[:, 0]-1)
  log_q_z += start_s * (1-term_s) * jnp.log(1-jnp.take(b0, sequences[:, 0]-1))

  # Terms due to e
  start, finish = sequences[:, 0], sequences[:, 0] + sequences[:, 1] - 1
  log_q_z += (1-term_s) * (
    jnp.take(b1_cumsum, finish) - jnp.take(b1_cumsum, start))
  log_q_z += reach_i * jnp.log(1-jnp.take(b1, finish))

  # Terms due to I
  start = sequences[:, 0] + sequences[:, 1]
  finish = sequences[:, 0] + sequences[:, 1] + sequences[:, 2] - 1
  log_q_z += reach_i * (
    jnp.take(b2_cumsum, finish) - jnp.take(b2_cumsum, start))
  log_q_z += reach_i * (1-term_i) * jnp.log(1-jnp.take(b2, finish))

  return log_q_z


def enumerate_log_q_values(
    params: Array,
    sequences: Array) -> Array:
  """Enumerate values of log_q for variational parameters."""
  a0, b0, b1, b2 = params[0], params[1], params[2], params[3]
  time_total = np.max(sequences)

  log_q_z = np.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * np.log(a0) + (1-start_s) * np.log(1-a0)
  log_q_z += start_s * (sequences[:, 0]-1)*np.log(b0)
  log_q_z += start_s * (1-term_s) * np.log(1-b0)

  # Terms due to e
  log_q_z += (1-term_s) * (sequences[:, 1] - 1) * np.log(b1)
  log_q_z += reach_i * np.log(1-b1)

  # Terms due to I
  log_q_z += reach_i * (sequences[:, 2]-1) * np.log(b2)
  log_q_z += reach_i * (1-term_i) * np.log(1-b2)

  return log_q_z


def enumerate_log_q_values_grad(
    params: Array,
    sequences: Array) -> np.ndarray:
  """Enumerate gradient of log_q w.r.t. variational parameters."""
  a0, b0, b1, b2 = params[0], params[1], params[2], params[3]
  time_total = np.max(sequences)

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to S
  grad_a0 = start_s / a0 - (1-start_s) / (1-a0)
  grad_b0 = start_s * (sequences[:, 0]-1) / b0 - start_s * (1-term_s) / (1-b0)

  # Terms due to E
  grad_b1 = (1-term_s) * (sequences[:, 1] - 1) / b1 - reach_i / (1-b1)

  # Terms due to I
  grad_b2 = reach_i * (sequences[:, 2]-1) / b2 - reach_i * (1-term_i) / (1-b2)

  return np.stack([grad_a0, grad_b0, grad_b1, grad_b2], axis=0)


def estimate_h(
    samples: Mapping[int, Union[List[int], Array]],
    num_users: int,
    num_time_steps: int) -> float:
  """Estimate h from samples (from Gibbs sampling)."""
  sample_array = np.stack(
    [samples[u] for u in range(num_users)],
    axis=0
  )

  num_users_included = np.sum(sample_array[:, 2] > 0)

  does_terminate = np.logical_and(
    sample_array[:, 2] > 0,
    np.sum(sample_array, axis=1) >= num_time_steps
  )
  num_terminate = np.sum(does_terminate)

  h_est = (num_users_included-num_terminate) / (
    np.sum(sample_array[:, 2])-num_terminate)

  # For a small number of users, h can be outside [0, 1]
  return np.clip(h_est, 0., 1.)


def calc_log_joint(
    samples: Dict[int, Array],
    observations: List[constants.Observation],
    contacts: List[constants.Contact],
    prior,  # (potential_sequences, log_A_start)
    num_users: int,
    num_time_steps: int,
    alpha: float,
    beta: float,
    probab_0: float,
    probab_channels: np.ndarray) -> np.ndarray:
  """Calculate the log joint probabilities from samples."""
  # Construct mapping to re-use log_A_start calculation
  log_A_start_mapping = dict(zip(*prior))

  seqs_sample = np.stack([samples[u] for u in range(num_users)], axis=0)

  log_c_z_u = calc_c_z_u(
    seqs_sample, observations, num_users=num_users, alpha=alpha, beta=beta)
  log_C = np.stack([log_c_z_u[u][u] for u in range(num_users)], axis=0)

  term_t = (seqs_sample[:, 0] >= num_time_steps).astype(np.float32)

  infect_counter = InfectiousContactCount(
    contacts=contacts,
    samples=samples,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  log_A = np.zeros((num_users))

  for user in range(num_users):
    seq_user = samples[user]
    log_A[user] += log_A_start_mapping[tuple(list(seq_user))]

    # Precompute p_u_t terms
    log_p_u_t = precompute_p_u_t(
      num_time_steps+1,  # Add 1 to remain 0-indexed
      user,
      infect_counter,
      samples,
      probab_channels
    )

    # Gather first terms of equation 17
    cumsum_log = np.cumsum(log_p_u_t)
    log_prod_p_u_t = cumsum_log[np.maximum(seq_user[0]-2, 0)]
    penalty = log_prod_p_u_t * (seq_user[0]-2 >= 0)

    log_p_not_get_infected = (log_p_u_t[np.maximum(seq_user[0]-1, 0)]
                              + np.log(1-probab_0))
    # Don't subtract log(p0) when terminate in I, as the term wasn't included
    # See `yield (t0, 0, 0)` in iter_prior() function
    penalty += (1-term_t[user])*(
      np.log(1 - np.exp(log_p_not_get_infected)) - np.log(probab_0))

    log_A[user] += penalty

  return np.sum(log_A + log_C)


def sigmoid(x):
  return 1/(1+np.exp(-x))


def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


def gen_dynamics_array(
    p0_val: float,
    g_val: float,
    h_val: float,
    num_time_steps: int):
  """Generate the dynamics log_q and log_q_tail.

  Follows notation of CRISP paper.
  """
  # Calculate the pmf of 'days spent' for each consecutive day
  q_e_vec = [0] + [g_val*(1-g_val)**(i-1) for i in range(1, num_time_steps+1)]
  q_i_vec = [0] + [h_val*(1-h_val)**(i-1) for i in range(1, num_time_steps+1)]

  # Normalise PMF
  pmf_e = np.array(q_e_vec) / np.sum(q_e_vec) + 1E-12
  pmf_i = np.array(q_i_vec) / np.sum(q_i_vec) + 1E-12

  # Find probabilities of termination for each consecutive day
  lqe, lqe_tail = q_log_tail_prob(pmf_e)
  lqi, lqi_tail = q_log_tail_prob(pmf_i)

  lqs = [k * np.log(1 - p0_val) + np.log(p0_val)
         for k in range(num_time_steps+1)]
  lqs_tail = [k * np.log(1 - p0_val) for k in range(num_time_steps+1)]
  return lqs, lqs_tail, lqe, lqe_tail, lqi, lqi_tail


def forward_propagate_q(params: Array, num_time_steps: int) -> Array:
  """Forward propagate state probabilities under q.

  Args:
    params: array of size 3*num_time_steps+1
    num_time_steps: number of time steps

  Returns:
    Array of size [num_time_steps, 4]

  """
  assert len(params) == (3*num_time_steps+1)

  params = np.array(params)
  a0 = params[0]
  b0 = params[1:num_time_steps+1]
  b1 = params[num_time_steps+1:2*num_time_steps+1]
  b2 = params[2*num_time_steps+1:3*num_time_steps+1]

  q_z = np.zeros((num_time_steps, 4))
  q_z[0] = np.array([a0, 1-a0, 0, 0])
  transitions = np.zeros((4, 4, num_time_steps))
  transitions[0][0], transitions[0][1] = b0, 1-b0
  transitions[1][1], transitions[1][2] = b1, 1-b1
  transitions[2][2], transitions[2][3] = b2, 1-b2
  transitions[3][3] = np.ones((num_time_steps))

  for i in range(1, num_time_steps):
    q_z[i] = transitions[:, :, i-1].T.dot(q_z[i-1])

  return q_z


def get_q_marginal_infected(params: Array, num_time_steps: int) -> Array:
  """Calculate marginal probility under q of being infected."""
  assert len(params) == (3*num_time_steps+1)

  return forward_propagate_q(params, num_time_steps)[:, 2]


def precompute_d_penalty_terms_vi(
    q_marginal_infected: Array,
    p0: float,
    p1: float,
    past_contacts: Iterable[Tuple[int, int, int]],
    num_time_steps: int) -> Tuple[Array, Array]:
  """Precompute penalty terms for Structured Variational Inference.

  Consider similarity to 'precompute_d_penalty_terms_fn' and how the log-term is
  applied."""
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term, d_no_term = np.zeros((num_time_steps+1)), np.zeros((num_time_steps+1))

  # Scales with O(T)
  for t_contact, contact_group in itertools.groupby(
      past_contacts, key=lambda x: x[0]):
    _, users, features = zip(*contact_group)
    features = list(features)
    np.testing.assert_array_almost_equal(features, 1.)

    # Scales with O(2**(C_max))
    probs = [q_marginal_infected[user][t_contact] for user in users]
    for num_infected, prob_total in it_num_infected_probs(probs):
      d_term[t_contact+1] += prob_total*np.log(1-(1-p0)*(1-p1)**num_infected)
      d_no_term[t_contact+1] += prob_total * num_infected * np.log(1-p1)

    # Subtract p0 at termination only once:
    d_term[t_contact+1] -= np.log(p0)

  # No termination (when t0 == num_time_steps) should not incur penalty
  d_term[num_time_steps] = 0.
  return d_term, d_no_term


def precompute_d_penalty_terms_fn_old(
    q_marginal_infected: Array,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[Array, Array]:
  """Precompute penalty terms for inference with Factorised Neighbors.
  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term, d_no_term = np.zeros((num_time_steps+1)), np.zeros((num_time_steps+1))

  # Scales with O(T)
  for t_contact, contact_group in itertools.groupby(
      past_contacts, key=lambda x: x[0]):
    if t_contact < 0:
      continue

    # TODO simplify
    contacts = [c[1] for c in contact_group]

    # Calculate in log domain to prevent underflow
    log_expectation = 0.

    # Scales with O(num_contacts)
    for user_contact in contacts:
      prob_infected = q_marginal_infected[user_contact][t_contact]
      log_expectation += np.log(prob_infected*(1-p1) + (1-prob_infected))

    d_no_term[t_contact+1] = log_expectation
    d_term[t_contact+1] = (
      np.log(1 - (1-p0)*np.exp(log_expectation)) - np.log(p0))

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term, d_no_term


@njit
def precompute_d_penalty_terms_fn(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[Array, Array]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1))
  d_no_term = np.zeros((num_time_steps+1))

  if len(past_contacts) == 0:
    return d_term, d_no_term

  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  t_contact = past_contacts[0][0]
  contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    if row[0] == t_contact:
      contacts.append(row[1])
    else:
      # Calculate in log domain to prevent underflow
      log_expectation = 0.

      # Scales with O(num_contacts)
      for user_contact in contacts:
        prob_infected = q_marginal_infected[user_contact][t_contact]
        log_expectation += np.log(prob_infected*(1-p1) + (1-prob_infected))

      d_no_term[t_contact+1] = log_expectation
      d_term[t_contact+1] = (
        np.log(1 - (1-p0)*np.exp(log_expectation)) - np.log(p0))

      # Reset loop stuff
      t_contact = row[0]
      contacts = [np.int32(x) for x in range(0)]

      if t_contact < 0:
        break

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term, d_no_term


def it_num_infected_probs(probs: List[float]) -> Iterable[Tuple[int, float]]:
  """Iterates over the number of infected neighbors and its probabilities.

  NOTE: this function scales exponential in the number of neighbrs,
  O(2**len(probs))

  Args:
    probs: List of floats, each being the probability of a neighbor being
    infected.

  Returns:
    iterator with tuples of the number of infected neighbors and its probability
  """
  for is_infected in itertools.product([0, 1], repeat=len(probs)):
    yield sum(is_infected), math.prod(
      abs(is_infected[i] - 1 + probs[i]) for i in range(len(probs)))


def spread_buckets(num_samples, num_buckets):
  assert num_samples >= num_buckets
  num_samples_per_bucket = (int(np.floor(num_samples / num_buckets))
                            * np.ones((num_buckets)))
  num_remaining = int(num_samples - np.sum(num_samples_per_bucket))
  num_samples_per_bucket[:num_remaining] += 1
  return num_samples_per_bucket


def maybe_make_dir(dirname: str):
  if not os.path.exists(dirname):
    logger.info(os.getcwd())
    logger.info(f"Making data_dir {dirname}")
    os.makedirs(dirname)


def quantize(message: Union[np.ndarray, float], num_levels: int
             ) -> Union[np.ndarray, float]:
  """Quantizes a message based on rounding.

  Numerical will be mid-bucket.

  TODO: implement quantization with coding scheme."""
  if num_levels < 0:
    return message

  if np.any(message > 1. + 1E-5):
    logger.info(np.min(message))
    logger.info(np.max(message))
    logger.info(np.mean(message))
    raise ValueError(f"Invalid message {message}")
  message = np.clip(message, 0., 1.-1E-9)
  message_at_floor = np.floor(message * num_levels) / num_levels
  return message_at_floor + (.5 / num_levels)


def quantize_floor(message: Union[np.ndarray, float], num_levels: int
                   ) -> Union[np.ndarray, float]:
  """Quantizes a message based on rounding.

  Numerical will be at the floor of the bucket.

  TODO: implement quantization with coding scheme."""
  if num_levels < 0:
    return message

  if np.any(message > 1. + 1E-5):
    logger.info(np.min(message))
    logger.info(np.max(message))
    logger.info(np.mean(message))
    raise ValueError(f"Invalid message {message}")
  return np.floor(message * num_levels) / num_levels


def get_cpu_count() -> int:
  # Divide cpu_count among tasks when running multiple tasks via SLURM
  num_tasks = 1
  if (slurm_ntasks := os.getenv("SLURM_NTASKS")):
    num_tasks = int(slurm_ntasks)
  return int(os.cpu_count() // num_tasks)


def normalize(x: Union[List[float], np.ndarray]):
  return x / np.sum(x)


def check_exists(filename: str):
  if not os.path.exists(filename):
    logger.warning(f"File does not exist {filename}, current wd {os.getcwd()}")
    raise FileNotFoundError(f"File does not exist {filename}")


def update_beliefs(
    belief_matrix: np.ndarray,
    belief_update: np.ndarray,
    user_slice: Union[List[Any], np.ndarray],
    users_stale: Optional[Union[np.ndarray, List[int]]] = None):
  """Updates the matrix of beliefs."""
  if users_stale is None:
    belief_matrix[user_slice] = belief_update
    return belief_matrix

  for user, bp_belief in zip(user_slice, belief_update):
    if user in users_stale:
      continue
    belief_matrix[user] = bp_belief
  return belief_matrix


def sample_stale_users(
    user_slice: Optional[np.ndarray]) -> Optional[np.ndarray]:
  if user_slice is None:
    return None
  np.random.shuffle(user_slice)
  return user_slice[:int(len(user_slice) // 2)]


def get_joblib_backend():
  """Determines a backend for joblib.

  When developing on local PC, then use 'threading'.
  On remote computers, use 'loky'.
  """
  if "carbon" in socket.gethostname():
    return "threading"
  return "loky"
