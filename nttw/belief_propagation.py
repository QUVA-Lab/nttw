"""Belief propagation for CRISP-like models."""
import datetime
import joblib
import numpy as np
from nttw import constants, logger, util
from scipy import special
import time
from typing import Any, Dict, List, Optional, Tuple, Union


def adjust_contacts(
    A_matrix: np.ndarray,
    p_infect_list: List[Union[float, np.ndarray]],
    p1: float):
  """Adjusts the transition matrix according to incoming infections."""
  if len(p_infect_list) == 0:
    return A_matrix
  A_adj = np.copy(A_matrix)

  # Do multiplication in log domain. May avoid numerical errors in case of many
  # contacts
  log_prob = np.log(A_adj[0][0])
  for p_inf in p_infect_list:
    log_prob += np.log((p_inf * (1-p1) + (1-p_inf) * 1))

  p_transition = np.exp(log_prob)
  A_adj[0][0] = p_transition
  A_adj[0][1] = 1-p_transition
  return A_adj


def adjust_matrices_map(
    A_matrix: np.ndarray,
    p1: float,
    forward_messages: Dict[Tuple[int, int], Any],
    num_time_steps: int) -> Tuple[
      List[np.ndarray], List[Tuple[int, int]]]:
  """Adjusts dynamics matrices based in messages from incoming contacts."""
  A_adjusted = []

  # First collate all incoming forward messages according to timestep
  user_time_backward = []
  p_lists = [[] for _ in range(num_time_steps)]
  for (user_backward, timestep), p_inf_message in forward_messages.items():
    p_lists[timestep].append(p_inf_message)
    user_time_backward.append((user_backward, timestep))

  # Then adjust matrices according to incoming infections
  for t_now in range(num_time_steps):
    # user_u is in second element of tuple
    A_adjusted.append(adjust_contacts(A_matrix, p_lists[t_now], p1=p1))
  return A_adjusted, user_time_backward


def forward_backward_user(
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    user: int,
    map_backward_message: List[Dict[Any, np.ndarray]],
    map_forward_message: List[Dict[Any, Union[float, np.ndarray]]],
    num_time_steps: int,
    obs_messages: np.ndarray,
    start_belief: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Any], List[Any]]:
  """Does forward backward step for one user.

  Args:
    user: integer which is the user index
    map_backward_mesage: for each user, this is a dictionary with keys
      (user_v, timestep), and the value the message, which is an array of length
      4 for each of the SEIR states.
    map_forward_message: for each user, this is a dictionary with keys
      (user_u, timestep), and the value is the probability this message assigns
      to user_u being infected at timestep t.
    start_belief: array/list of length 4, being start belief for SEIR states.

  Returns:
    marginal beliefs for this user after running bp, and the forward and
    backward messages that this user sends out.
  """
  # Default to start_belief as 1.-p0 in S and p0 in E
  if start_belief is None:
    start_belief = np.array([1.-p0, p0, 0., 0.])
  assert start_belief.shape == (4,), f"Shape {start_belief.shape} is not [4] "

  # Collate backward messages
  user_time_forward = []  # Collect the users that need a forward message
  mu_back_contact_log = np.zeros((num_time_steps, 4))  # Collate in matrix
  for (user_forward, timestep), message in map_backward_message[user].items():
    user_time_forward.append((user_forward, timestep))
    mu_back_contact_log[timestep] += np.log(message + 1E-12)
  mu_back_contact_log -= special.logsumexp(
    mu_back_contact_log, axis=1, keepdims=True)
  mu_back_contact = np.exp(mu_back_contact_log)

  # Clip messages in case of quantization
  mu_back_contact = np.clip(mu_back_contact, 0.0001, 0.9999)

  # TODO remove outer list
  mu_f2v_forward = [None for _ in range(num_time_steps)]
  mu_f2v_backward = [None for _ in range(num_time_steps)]

  # Forward messages can be interpreted as modifying the dynamics matrix.
  # Therefore, we precompute these matrices using the incoming forward messages
  A_user, user_time_backward = adjust_matrices_map(
    A_matrix, p1, map_forward_message[user], num_time_steps)

  betas = [None for _ in range(num_time_steps)]
  # Move all messages forward
  mu_f2v_forward[0] = start_belief
  for t_now in range(1, num_time_steps):
    mu_f2v_forward[t_now] = A_user[t_now-1].T.dot(
      mu_f2v_forward[t_now-1] * obs_messages[t_now-1]
      * mu_back_contact[t_now-1])

  # Move all messages backward
  mu_f2v_backward[num_time_steps-1] = np.ones((4))
  for t_now in range(num_time_steps-2, -1, -1):
    mu_f2v_backward[t_now] = A_user[t_now].dot(
      mu_f2v_backward[t_now+1] * obs_messages[t_now+1]
      * mu_back_contact[t_now+1])

  # Collect marginal beliefs
  for t_now in range(num_time_steps):
    # TODO: do in logspace
    betas[t_now] = (mu_f2v_forward[t_now] * mu_f2v_backward[t_now]
                    * obs_messages[t_now] * mu_back_contact[t_now])

  # Calculate messages backward
  messages_back = []
  for user_backward, timestep_back in user_time_backward:
    assert timestep_back >= 0, (
      "Cannot send a message back on timestep 0. "
      "Probably some contact was defined for -1?")
    A_back = A_user[timestep_back]

    # This is the term that needs cancelling due to the forward message
    p_forward = map_forward_message[user][(user_backward, timestep_back)]
    p_transition = A_back[0][0] / (p_forward * (1-p1) + (1-p_forward) * 1)

    # Cancel the terms in the two dynamics matrices
    A_back_0 = np.copy(A_back)
    A_back_1 = np.copy(A_back)
    A_back_0[0][0] = p_transition  # S --> S
    A_back_0[0][1] = 1. - p_transition  # S --> E
    A_back_1[0][0] = p_transition * (1-p1)  # S --> S
    A_back_1[0][1] = 1. - p_transition * (1-p1)  # S --> E

    # Calculate the SER terms and calculate the I term
    mess_SER = np.inner(
      A_back_0.dot(mu_f2v_backward[timestep_back+1]
                   * obs_messages[timestep_back+1]),
      mu_f2v_forward[timestep_back] * obs_messages[timestep_back])
    mess_I = np.inner(
      A_back_1.dot(mu_f2v_backward[timestep_back+1]
                   * obs_messages[timestep_back+1]),
      mu_f2v_forward[timestep_back] * obs_messages[timestep_back])
    message_back = np.array([mess_SER, mess_SER, mess_I, mess_SER]) + 1E-12
    message_back /= np.sum(message_back)
    if np.any(np.logical_or(np.isinf(message_back), np.isnan(message_back))):
      logger.debug(f"Message back: \n {message_back}")
      logger.debug(f"mu_back_contact: \n {mu_back_contact}")
    messages_back.append((user, user_backward, timestep_back, message_back))

  # Calculate messages forward
  messages_forward = []
  for user_forward, timestep in user_time_forward:
    assert timestep < num_time_steps, (
      "Cannot send a message back on timestep <num_time_steps>. "
      "Probably some contact was defined for <num_time_steps>?")
    message_backslash = util.normalize(
      map_backward_message[user][(user_forward, timestep)] + 1E-12)
    # TODO: do in logspace
    message = (betas[timestep] / message_backslash)
    message /= np.sum(message)
    if np.any(np.logical_or(np.isinf(message), np.isnan(message))):
      logger.debug(f"Message forward: \n {message}")
      logger.debug(f"Betas: \n {betas[timestep]}")
      logger.debug(f"mu_back_contact: \n {mu_back_contact[timestep]}")
      logger.debug(f"mu_f2v_forward: \n {mu_f2v_forward[timestep]}")
      logger.debug(f"mu_f2v_backward: \n {mu_f2v_backward[timestep]}")
      logger.debug(f"obs_messages: \n {obs_messages[timestep]}")
      logger.debug(f"Backward backslash: \n {message_backslash}")
      logger.debug(f"mu_f2v_forward: \n {mu_f2v_forward}")
      raise ValueError("NaN or INF in message")
    messages_forward.append((user, user_forward, timestep, message[2]))

  bp_beliefs_user = np.stack([
    betas[i] for i in range(num_time_steps)], axis=0)
  bp_beliefs_user /= np.sum(bp_beliefs_user, axis=1, keepdims=True)
  return bp_beliefs_user, messages_back, messages_forward


def do_backward_forward_subset(
    user_subset: List[int],
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    num_time_steps: int,
    obs_messages: np.ndarray,
    map_backward_message: List[Dict[Any, Union[float, np.ndarray]]],
    map_forward_message: List[Dict[Any, Union[float, np.ndarray]]],
    start_beliefs: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray, List[Any], List[Any]]:
  """Does forward backward on a subset of users in sequence.

  Note, the messages are appended, and thus not updated between users!
  """
  tstart = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
  t0 = time.time()
  do_timing = False

  bp_beliefs_subset = np.zeros((len(user_subset), num_time_steps, 4))

  messages_backward_subset, messages_forward_subset = [], []
  for i, user_run in enumerate(user_subset):
    start_belief = start_beliefs[i] if start_beliefs is not None else None
    bp_beliefs_subset[i], messages_back_user, messages_forward_user = (
      forward_backward_user(
        A_matrix, p0, p1, user_run, map_backward_message, map_forward_message,
        num_time_steps, obs_messages[i], start_belief))
    messages_backward_subset += messages_back_user
    messages_forward_subset += messages_forward_user

  if do_timing:
    logger.info(f"BP passes start at {tstart}, {time.time()-t0:9.1f} seconds")
  return (user_subset, bp_beliefs_subset, messages_forward_subset,
          messages_backward_subset)


def update_message(
    message_old: np.ndarray, message_in: np.ndarray,
    damping: float):
  assert 0.0 <= damping <= 1.0
  return damping*message_old + (1-damping)*message_in


def do_backward_forward_and_message(
    A_matrix: np.ndarray,
    p0: float,
    p1: float,
    num_time_steps: int,
    obs_messages: np.ndarray,
    num_users: int,
    map_backward_message: List[Dict[Any, Union[float, np.ndarray]]],
    map_forward_message: List[Dict[Any, Union[float, np.ndarray]]],
    start_belief: Optional[np.ndarray] = None,
    damping: float = 0.0,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    freeze_backwards: bool = False,
    parallel: Optional[joblib.Parallel] = None,
    num_jobs: int = 1,
    verbose: int = 0,
    ) -> Tuple[np.ndarray, List[Dict[Any, Any]], List[Dict[Any, Any]]]:
  """Runs forward and backward messages for one user and collates messages."""

  users_stale_now = util.sample_stale_users(users_stale)

  # Spread the users for this slice over the parallel job(s)
  num_users_per_job = util.spread_buckets(num_users, num_jobs)
  slices = np.concatenate(([0], np.cumsum(num_users_per_job))).astype(np.int32)
  assert slices[-1] == num_users
  if verbose:
    logger.debug(f"Slices {slices}")

  if parallel is None:
    # Parallel with 1 job implies no parallelism
    parallel = joblib.Parallel(n_jobs=1)

  user_slices = [
    list(range(slices[n_job], slices[n_job+1])) for n_job in range(num_jobs)]

  obs_messages_slices = [
    np.stack([obs_messages[user] for user in user_slice], axis=0)
    for user_slice in user_slices]

  start_belief_slices = [None for _ in range(num_jobs)]
  if start_belief is not None:
    start_belief_slices = [
      start_belief[user_slice] for user_slice in user_slices]

  results = parallel(joblib.delayed(do_backward_forward_subset)(
    user_slice,
    A_matrix=A_matrix,
    p0=p0,
    p1=p1,
    num_time_steps=num_time_steps,
    obs_messages=obs_messages_slices[num_slice],
    map_backward_message={
      u: map_backward_message[u] for u in user_slice},
    map_forward_message={
      u: map_forward_message[u] for u in user_slice},
    start_beliefs=start_belief_slices[num_slice]
  ) for num_slice, user_slice in enumerate(user_slices))

  bp_beliefs_matrix = np.zeros((num_users, num_time_steps, 4))
  for result in results:
    (user_subset, bp_beliefs_subset, messages_forward_subset,
      messages_backward_subset) = result

    # Put beliefs in big matrix
    bp_beliefs_matrix = util.update_beliefs(
      bp_beliefs_matrix, bp_beliefs_subset, user_subset, users_stale_now)

    if users_stale_now is not None:
      bp_beliefs_matrix[users_stale_now] = .25 * np.ones((num_time_steps, 4))

    # Put backward messages generated by user_run in hashmap
    for user_send, user_backward, tstep, message in messages_backward_subset:
      if (users_stale_now is not None) and user_send in users_stale_now:
        continue
      message_new = update_message(
        map_backward_message[user_backward][(user_send, tstep)],
        message,
        damping)
      if freeze_backwards:
        message_new = np.ones((4)) / 4.
      message_quant = util.quantize(message_new, quantization)
      map_backward_message[user_backward][(user_send, tstep)] = message_quant

    # Put forward messages generated by user_run in hashmap
    for user_send, user_forward, tstep, message in messages_forward_subset:
      if (users_stale_now is not None) and user_send in users_stale_now:
        continue
      message_new = update_message(
        map_forward_message[user_forward][(user_send, tstep)],
        message,
        damping)
      message_quant = util.quantize(message_new, quantization)
      map_forward_message[user_forward][(user_send, tstep)] = message_quant

  np.testing.assert_array_almost_equal(
    np.sum(bp_beliefs_matrix, axis=-1), 1., decimal=3)

  return bp_beliefs_matrix, map_backward_message, map_forward_message


def init_message_maps(contacts_all: List[constants.Contact], num_users: int):
  """Initialises the message maps."""
  # Put backward messages in hashmap, such that they can be overwritten when
  #   doing multiple iterations in loopy belief propagation
  map_backward_message = [{} for _ in range(num_users)]
  map_forward_message = [{} for _ in range(num_users)]

  for contact in contacts_all:
    # Think about contact u --> v
    # Then backward messages are a list indexed in u,
    #   with a hashmap with keys (v,time) and value a distribution over z_{u,t}
    # Then forward messages are a list indexed in v,
    #   with a hashmap with keys (u,time) and value the (approx) probability of
    #   infection of z_u at time t
    map_backward_message[
      contact[0]][(contact[1], contact[2])] = np.ones((4))
    map_forward_message[
      contact[1]][(contact[0], contact[2])] = 0.
  return map_forward_message, map_backward_message
