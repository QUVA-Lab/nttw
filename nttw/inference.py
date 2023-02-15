"""Inference methods for contact-graphs."""
import datetime
import joblib
import itertools
import jax  # pylint: disable=unused-import
from jax.example_libraries import optimizers
import jax.numpy as jnp
from nttw import constants, logger, util
import numba
import numpy as np
import os
import random
from scipy import special
import time
import tqdm
from typing import Any, Dict, List, Optional, Tuple


def gibbs_sampling(  # pylint: disable=dangerous-default-value
    num_users: int,
    g_param: float,
    h_param: float,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_channels: np.ndarray,
    probab_0: float = 0.001,
    alpha: float = 0.001,
    beta: float = 0.01,
    num_samples: int = 1000,
    skip_steps: int = 1,
    num_burn: int = 100,
    start_belief: Optional[np.ndarray] = None,
    do_learning: bool = False,
    verbose: bool = False,
    diagnostic: Optional[Any] = None):
  """Gibbs sampling to infer latent SEIR states."""
  assert start_belief is None, "Not implemented yet"
  time_start = time.time()

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=True)))

  # If 'start_belief' is provided, the prior will be applied per user, later
  if start_belief is None:
    prior = [1-probab_0, probab_0, 0., 0.]
  else:
    prior = [.25, .25, .25, .25]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab_0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  samples_acc = {user: np.zeros((3)) for user in range(num_users)}

  num_updates = (num_samples + num_burn) * skip_steps
  acc_trajectory_user = np.zeros((num_users, num_samples, 3))
  grad_p1 = np.zeros((num_samples, num_users))
  estimates_h = np.zeros((num_samples))
  log_joints = np.zeros((num_samples))

  # Precompute log_c_z_u[user]
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)

  term_t = (seq_array[:, 0] >= num_time_steps).astype(np.float32)

  time_calc_A = 0.
  time_calc_B = 0.
  time_calc_B_pre = 0.
  time_calc_sample = 0.
  time_do_acc = 0.
  time_calc_joint = 0.

  # Initialize samples
  samples_current = {
    # Being: t0, de, di; time in Susceptible, Exposed, and Infected
    user: np.array([num_time_steps//3, num_time_steps//3, num_time_steps//3])
    for user in range(num_users)}
  if verbose:
    print(f"Initialize samples at {samples_current}")

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=samples_current,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  # Start taking Gibbs samples
  for num_update in range(num_updates):
    user_list = list(range(num_users))
    random.shuffle(user_list)
    for user in user_list:
      # Remove trace from this user. Otherwise there's overcounting of
      # infectious contacts in equation 21.
      infect_counter.update_infect_count(
        user, samples_current[user], remove=True)

      log_A = np.copy(log_A_start)

      # Apply start_belief
      if start_belief is not None:
        log_A += np.log(
          util.enumerate_start_belief(seq_array, start_belief[user]))

      t0 = time.time()
      # Precompute p_u_t terms
      log_p_u_t = util.precompute_p_u_t(
        num_time_steps+1,  # Add 1 to remain 0-indexed
        user,
        infect_counter,
        samples_current,
        probab_channels)

      # Gather first terms of equation 17
      cumsum_log = np.cumsum(log_p_u_t)
      log_prod_p_u_t = np.take(cumsum_log, seq_array[:, 0]-2)
      log_A += log_prod_p_u_t * (seq_array[:, 0]-2 >= 0)

      log_p_not_get_infected = (np.take(log_p_u_t, seq_array[:, 0]-1)
                                + np.log(1-probab_0))
      # Don't subtract log(p0) when terminate in I, as the term wasn't included
      # See `yield (t0, 0, 0)` in iter_prior() function
      log_A += (1-term_t)*(
        np.log(1 - np.exp(log_p_not_get_infected)) - np.log(probab_0))
      time_calc_A += time.time()-t0

      t0 = time.time()
      log_b_ratio = util.precompute_b_z(
        num_time_steps, user, infect_counter, samples_current,
        probab_channels, probab_0)
      time_calc_B_pre += time.time() - t0

      # Calculating B
      t0 = time.time()
      b_log_sum = np.cumsum(log_b_ratio)

      di_start = seq_array[:, 0] + seq_array[:, 1]
      di_end = seq_array[:, 0] + seq_array[:, 1] + seq_array[:, 2]

      log_B = (np.take(b_log_sum, di_end-1)
               - np.take(b_log_sum, di_start-1))
      # print(f"{np.max(log_B):8.3f}{np.max(log_B):8.3f}")
      # print(f"{np.min(log_B):8.3f}{np.min(log_B):8.3f}")
      time_calc_B += time.time() - t0

      t0 = time.time()
      # Set log_sampling_prob
      log_sampling_probs = log_A + log_B + log_c_z_u[user]
      if np.any(np.isinf(log_sampling_probs)):
        print(
          (f"Infinity encountered"
           f"at value {log_sampling_probs}"
           f"at \nlog_A {log_A}, \nlog_B {log_B}, "
           f"\nlog_C{log_c_z_u[user]}"))

      # Sample from log probabilities
      probs = np.exp(log_sampling_probs - np.max(log_sampling_probs))
      probs_cumsum_unnormalised = np.cumsum(probs)
      probs_cumsum = probs_cumsum_unnormalised / probs_cumsum_unnormalised[-1]
      num_seq_sampled = np.argmax(probs_cumsum >= random.random())

      # Set Gibbs sample
      trace_new = seq_array[num_seq_sampled]
      samples_current[user] = trace_new
      time_calc_sample += time.time() - t0

      infect_counter.update_infect_count(user, trace_new)

    t0 = time.time()
    # Update accumulators after burn in
    if num_update >= (num_burn * skip_steps) and (num_update % skip_steps == 0):
      num_sample = (num_update // skip_steps) - num_burn

      if do_learning:
        t_joint = 0
        log_joint = util.calc_log_joint(
          samples_current,
          observations_all,
          contacts_all,
          prior=(seq_array, log_A_start),
          num_users=num_users,
          num_time_steps=num_time_steps,
          alpha=alpha,
          beta=beta,
          probab_0=probab_0,
          probab_channels=probab_channels)
        log_joints[num_sample] = log_joint
        time_calc_joint += time.time() - t_joint

      for user in range(num_users):
        acc_trajectory_user[user][num_sample] = samples_current[user]
        samples_acc[user] += samples_current[user]

        if do_learning:

          # TODO make this below function calls efficient for multiple users
          num_infected_preparents = util.gather_infected_precontacts(
            num_time_steps=num_time_steps,
            samples_current=samples_current,
            past_contacts=infect_counter.get_past_contacts(user=user),
          )

          grad_p1[num_sample, user] = util.calc_grad_p1(
            num_infected_preparents,
            probab_0=probab_0,
            probab_1=probab_channels[0],
            time_s=samples_current[user][0]
          )

          estimates_h[num_sample] = util.estimate_h(
            samples_current, num_users, num_time_steps)

    time_do_acc += time.time() - t0

  if verbose:
    print(f"Time spent on A     {time_calc_A:8.2f}")
    print(f"Time spent on B_pre {time_calc_B_pre:8.2f}")
    print(f"Time spent on B     {time_calc_B:8.2f}")
    print(f"Time spent on samp  {time_calc_sample:8.2f}")
    print(f"Time spent on acc   {time_do_acc:8.2f}")

    print((f"Sampled {num_samples} samples in "
           f"{time.time() - time_start:.1f} seconds"))

  if diagnostic:
    diagnostic.log({
      "time_A": time_calc_A,
      "time_B": time_calc_B,
      }, commit=False)

  if do_learning:
    return acc_trajectory_user, grad_p1, log_joints, estimates_h
  return acc_trajectory_user


def struc_var_inf(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    alpha: float = 0.001,
    beta: float = 0.01,
    num_updates: int = 1000,
    learning_rate: float = 0.5,
    momentum: float = 0.5,
    verbose: bool = False):
  """Structural Variational Inference to infer latent SEIR states."""
  if len(contacts_all) > 0:
    print("Warning: only implemented for no contacts")

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  q_logit = np.zeros((4))
  num_params = len(q_logit)
  grad_moment = np.zeros((num_params))

  ELBO = None

  for num_update in range(num_updates):
    grads = np.zeros((num_params))
    ELBO = np.array([0.])

    # Collect gradients over all users.
    # Assumes that no contacts exist between users
    q_param = util.sigmoid(q_logit)
    log_q_vec = util.enumerate_log_q_values(q_param, seq_array)
    q_vec = np.exp(log_q_vec)
    np.testing.assert_almost_equal(np.sum(q_vec), 1., decimal=4,
                                   err_msg=f'Found sum {np.sum(q_vec)}')
    grad_q_vec = (util.enumerate_log_q_values_grad(q_param, seq_array)
                  * q_vec)  # type: ignore

    for user in range(num_users):
      log_joint = log_c_z_u[user] + log_A_start
      # import pdb; pdb.set_trace()

      # Multiply with
      likelihood = q_vec.dot(log_joint)
      entropy = -q_vec.dot(log_q_vec)
      ELBO += likelihood + entropy

      likelihood_grads = grad_q_vec.dot(log_joint)
      entropy_grads = grad_q_vec.dot(-(1 + log_q_vec))
      grads += likelihood_grads + entropy_grads

      # Store statistics

    grad_moment = momentum*grad_moment + (1-momentum)*grads
    q_logit = q_logit + learning_rate * grad_moment

    if verbose:
      with jax.numpy.printoptions(precision=2, suppress=True):
        print((f"At {num_update:5} "
               f"New beta params: {util.sigmoid(q_logit)} "
               f"at ELBO {ELBO}"))

  q_param = util.sigmoid(q_logit)
  return q_param, np.array(ELBO)


def check_sum_1(vec):
  try:
    np.testing.assert_almost_equal(np.sum(vec), 1., decimal=4,
                                   err_msg=f'Found sum {np.sum(vec)}')
  except AssertionError:
    import pdb; pdb.set_trace()  # pylint: disable=import-outside-toplevel,multiple-statements,forgotten-debug-statement


def unstruc_var_inf_from_posterior(
    log_posterior: np.ndarray,
    num_users: int) -> Tuple[np.ndarray, np.ndarray]:
  """Unstructured variational inference when true posterior is available."""

  assert num_users == 2

  num_sequences_prod = len(log_posterior)
  num_sequences = np.sqrt(num_sequences_prod).astype(np.int32)

  # Initialize q distributions
  q_0 = util.softmax(np.random.randn(num_sequences))
  q_1 = util.softmax(np.random.randn(num_sequences))

  log_p_square = np.reshape(log_posterior, (num_sequences, num_sequences))
  p_square = np.exp(log_p_square)

  for _ in range(10):
    q_1_unnorm = log_p_square.T.dot(q_0)
    q_1 = util.softmax(q_1_unnorm)

    q_0_unnorm = log_p_square.dot(q_1)
    q_0 = util.softmax(q_0_unnorm)

  q_reverse = np.outer(q_0, q_1).flatten()

  # Minimize with forward KL
  p_vec_0 = np.sum(p_square, axis=1)
  p_vec_1 = np.sum(p_square, axis=0)
  q_forward = np.outer(p_vec_0, p_vec_1).flatten()

  return q_reverse, q_forward


def unstruc_var_inf_from_posterior3(
    log_posterior: np.ndarray,
    num_users: int) -> Tuple[np.ndarray, np.ndarray]:
  """Unstructured variational inference when true posterior is available."""

  assert num_users == 3

  num_sequences_prod = len(log_posterior)
  num_sequences = np.round(np.power(num_sequences_prod, 1/3)).astype(np.int32)

  # Initialize q distributions
  q_0 = util.softmax(np.random.randn(num_sequences))
  q_1 = util.softmax(np.random.randn(num_sequences))
  q_2 = util.softmax(np.random.randn(num_sequences))

  log_p_square = np.reshape(
    log_posterior, (num_sequences, num_sequences, num_sequences))
  p_square = np.exp(log_p_square)

  for _ in range(10):
    q_2_unnorm = np.transpose(log_p_square, [2, 1, 0]).dot(q_0).dot(q_1)
    q_2 = util.softmax(q_2_unnorm)

    q_1_unnorm = np.transpose(log_p_square, [1, 2, 0]).dot(q_0).dot(q_2)
    q_1 = util.softmax(q_1_unnorm)

    q_0_unnorm = log_p_square.dot(q_2).dot(q_1)
    q_0 = util.softmax(q_0_unnorm)

  q_reverse = np.einsum('i,j,k->ijk', q_0, q_1, q_2).flatten()

  # Minimize with forward KL
  p_vec_0 = np.sum(np.sum(p_square, axis=2), axis=1)
  p_vec_1 = np.sum(np.sum(p_square, axis=2), axis=0)
  p_vec_2 = np.sum(np.sum(p_square, axis=1), axis=0)
  q_forward = np.einsum('i,j,k->ijk', p_vec_0, p_vec_1, p_vec_2).flatten()

  return q_reverse, q_forward


@numba.njit
def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit
def fn_step_wrapped(
    user_slice: np.ndarray,
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    past_contacts_array: np.ndarray,
    start_belief: Optional[np.ndarray] = None):
  """Wraps one step of Factorised Neighbors over a subset of users.

  Args:
    user_slice: list of user id for this step
    seq_array_hot: array in [num_time_steps, 4, num_sequences]
    log_c_z_u: array in [len(user_slice), num_sequences], C-terms according to
      CRISP paper
    log_A_start: array in [num_sequences], A-terms according to CRISP paper
    p_infected_matrix: array in [num_users, num_time_steps]
    num_time_steps: number of time steps
    probab0: probability of transitioning S->E
    probab1: probability of transmission given contact
    past_contacts: iterator with elements (timestep, user_u, features)
    start_belief: matrix in [len(user_slice), 4], i-th row is assumed to be the
      start_belief of user user_slice[i]
  """
  with numba.objmode(t0='f8'):
    t0 = time.time()

  post_exps = np.zeros((len(user_slice), num_time_steps, 4))
  num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int64)

  assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
    "seq_array_hot is expected as one-hot array")

  seq_array_hot = seq_array_hot.astype(np.float64)
  num_sequences = seq_array_hot.shape[2]

  # Numba dot only works on float arrays
  states = np.arange(4, dtype=np.float64)
  state_start = seq_array_hot[0].T.dot(states).astype(np.int16)

  for i in range(len(user_slice)):

    d_term, d_no_term = util.precompute_d_penalty_terms_fn(
      p_infected_matrix,
      p0=probab0,
      p1=probab1,
      past_contacts=past_contacts_array[i],
      num_time_steps=num_time_steps)
    d_noterm_cumsum = np.cumsum(d_no_term)

    d_penalties = (
      np.take(d_noterm_cumsum, np.maximum(num_days_s-1, 0))
      + np.take(d_term, num_days_s))

    # Apply local start_belief
    start_belief_enum = np.zeros((num_sequences))
    if start_belief is not None:
      start_belief_enum = np.take(start_belief[i], state_start)
      start_belief_enum = np.log(start_belief_enum + 1E-12)
      assert start_belief_enum.shape == log_A_start.shape

    # Numba only does matmul with 2D-arrays, so do reshaping below
    log_joint = softmax(
      log_c_z_u[i] + log_A_start + d_penalties + start_belief_enum)
    post_exps[i] = np.reshape(np.dot(
      seq_array_hot.reshape(num_time_steps*4, num_sequences), log_joint),
      (num_time_steps, 4))

  with numba.objmode(t1='f8'):
    t1 = time.time()
  return user_slice, post_exps, t0, t1


def fact_neigh(
    num_users: int,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
    start_belief: Optional[np.ndarray] = None,
    alpha: float = 0.001,
    beta: float = 0.01,
    damping: float = 0.0,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    num_updates: int = 1000,
    verbose: bool = False,
    num_jobs: int = 8,
    trace_dir: Optional[str] = None,
    diagnostic: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
  """Inferes latent states using Factorised Neighbor method.

  Uses Factorised Neighbor approach from
  'The dlr hierarchy of approximate inference, Rosen-Zvi, Jordan, Yuille, 2012'

  Args:
    num_users: Number of users to infer latent states
    num_time_steps: Number of time steps to infer latent states
    observations_all: List of all observations
    contacts_all: List of all contacts
    probab_0: Probability to be infected spontaneously
    probab_1: Probability of transmission given contact
    g_param: float, dynamics parameter, p(E->I|E)
    h_param: float, dynamics parameter, p(I->R|I)
    start_belief: array in [num_users, 4], which are the beliefs for the start
      state
    alpha: False positive rate of observations, (1 minus specificity)
    beta: False negative rate of observations, (1 minus sensitivity)
    damping: number between 0 and 1 to damp messages. 0 corresponds to no
      damping, number close to 1 correspond to high damping.
    quantization: number of levels for quantization. Negative number indicates
      no use of quantization.
    num_updates: Number of rounds to update using Factorised Neighbor algorithm
    verbose: set to true to get more verbose output
    num_jobs: Number of jobs to use for parallelisation. Recommended to set to
      number of cores on your machine.

  Returns:
    array in [num_users, num_timesteps, 4] being probability of over
    health states {S, E, I, R} for each user at each time step
  """
  t_start_preamble = time.time()
  assert num_jobs <= num_users, "Cannot run more parallel jobs than users"

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  # If 'start_belief' is provided, the prior will be applied per user, later
  if start_belief is None:
    prior = [1-probab_0, probab_0, 0., 0.]
  else:
    prior = [.25, .25, .25, .25]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab_0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  # Precompute log(C) terms, relating to observations
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)

  q_marginal_infected = np.zeros((num_users, num_time_steps))
  q_marginal_acc = np.zeros((num_updates+1, num_users, num_time_steps, 4))
  post_exp = np.zeros((num_users, num_time_steps, 4))

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  # Parellelise one inference step over users.
  # Split users among number of jobs
  num_users_per_job = util.spread_buckets(num_users, num_jobs)
  slices = np.concatenate(([0], np.cumsum(num_users_per_job))).astype(np.int64)
  assert slices[-1] == num_users
  if verbose:
    print("Slices", slices)

  user_slices = [
    list(range(slices[n_job], slices[n_job+1])) for n_job in range(num_jobs)]
  log_c_z_u_s = [
    np.stack([log_c_z_u[user] for user in user_slice], axis=0)
    for user_slice in user_slices]

  start_belief_slices = [None for _ in range(num_jobs)]
  if start_belief is not None:
    start_belief_slices = [
      start_belief[user_slice] for user_slice in user_slices]

  logger.info(
    f"Parallelise FN with {num_jobs} jobs "
    f"after {time.time() - t_start_preamble:.1f} seconds on preamble")
  backend = util.get_joblib_backend()
  with joblib.Parallel(n_jobs=num_jobs, backend=backend) as parallel:
    for num_update in range(num_updates):
      # logger.info(f"Update {num_update}")

      # Sample stale users
      users_stale_now = util.sample_stale_users(users_stale)

      results = parallel(joblib.delayed(fn_step_wrapped)(
        np.array(user_slice),
        seq_array_hot,
        log_c_z_u_s[num_slice],
        log_A_start,
        q_marginal_infected,
        num_time_steps,
        probab_0,
        probab_1,
        infect_counter.get_past_contacts_slice(user_slice),
        start_belief_slices[num_slice],
      ) for num_slice, user_slice in enumerate(user_slices))

      for (user_slice, post_exp_users, tstart, tend) in results:
        if verbose:
          tstart_fmt = datetime.datetime.fromtimestamp(tstart).strftime(
            "%Y.%m.%d_%H:%M:%S")
          logger.info(f'Started on {tstart_fmt}, for {tend-tstart:12.1f}')

        post_exp = util.update_beliefs(
          post_exp, post_exp_users, user_slice, users_stale_now)

      # Collect statistics
      damping_use = damping if num_update > 0 else 0.0
      q_marginal_infected = (damping_use * q_marginal_infected
                             + (1-damping_use) * post_exp[:, :, 2])
      q_marginal_acc[num_update+1] = post_exp

      # Quantization
      if quantization > 0:
        q_marginal_infected = util.quantize_floor(
          q_marginal_infected, num_levels=quantization)

      if trace_dir:
        fname = os.path.join(trace_dir, f"trace_{num_update:05d}.npy")
        with open(fname, 'wb') as fp:
          np.save(fp, post_exp)

      if verbose:
        with np.printoptions(precision=2, suppress=True):
          print(q_marginal_infected[0])
          if num_users > 2:
            print(q_marginal_infected[2])
          print()
      if diagnostic:
        diagnostic.log({'user0': post_exp[0][:, 2].tolist()}, commit=False)
  return post_exp, q_marginal_acc


def unstruc_var_inf_jax(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01,
    num_updates: int = 1000,
    learning_rate: float = 0.5,
    momentum: float = 0.5,
    verbose: bool = False,
    do_factorize: bool = False):
  """Unstructured Variational Inference to infer SEIR states using JAX."""
  if len(contacts_all) > 0:
    print("Warning: only implemented for no contacts")
  del momentum

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  num_sequences = len(potential_sequences)
  log_A_start = np.array(log_A_start_list)

  def learning_rate_func(step):
    mult_exp = int(1.2 * step/num_updates)
    mult = 10**(-mult_exp)
    return learning_rate * mult

  assert num_users == 2

  # Find all sequences
  seq_product = list(itertools.product(potential_sequences, repeat=num_users))
  num_sequences_product = len(seq_product)
  print(f"Work with sequence product of length {num_sequences_product}")

  if do_factorize:
    def q_func(logits):
      q_distr0 = jax.nn.softmax(logits[:num_sequences])
      q_distr1 = jax.nn.softmax(logits[num_sequences:])
      q_distr = jnp.outer(q_distr0, q_distr1).flatten()
      return q_distr
  else:
    def q_func(logits):
      return jax.nn.softmax(logits)

  grad_q_func = jax.jacfwd(q_func)

  if do_factorize:
    num_params = 2*num_sequences
  else:
    num_params = num_sequences_product

  opt_init, opt_update, opt_get = optimizers.adam(
    learning_rate_func)
  # opt_init, opt_update, opt_get = optimizers.nesterov(
  #   learning_rate_func, .5)
  # opt_init, opt_update, opt_get = optimizers.sgd(
  #   learning_rate_func)
  opt_state = opt_init(
    jnp.array(np.random.randn(num_params)))
  q_logit = opt_get(opt_state)

  ELBO_array = np.zeros((num_updates))
  logits_array = np.zeros((num_updates, num_params))

  # Calculate array of log joints:
  log_joints = np.zeros((num_sequences_product))

  for num_sample, samples in enumerate(seq_product):
    samples_current = {u: samples[u] for u in range(num_users)}
    log_joints[num_sample] = util.calc_log_joint(
      samples=samples_current,
      observations=observations_all,
      contacts=contacts_all,
      prior=(potential_sequences, log_A_start),
      num_users=num_users,
      num_time_steps=num_time_steps,
      alpha=alpha,
      beta=beta,
      probab_0=probab_0,
      probab_channels=np.array([probab_1]))

  for num_update in tqdm.trange(num_updates):
    q_param = q_func(q_logit)

    log_likelihood = q_param.dot(log_joints)
    entropy = -q_param.dot(jnp.log(q_param))

    ELBO_array[num_update] = log_likelihood + entropy
    logits_array[num_update] = np.array(q_logit)

    grad_q_vec = grad_q_func(q_logit).T
    likelihood_grads = grad_q_vec.dot(log_joints)
    entropy_grads = grad_q_vec.dot(-(1 + jnp.log(q_param)))

    # -1 to maximise ELBO
    grads = -1*(likelihood_grads + entropy_grads)

    opt_state = opt_update(num_update, grads, opt_state)

    q_logit = opt_get(opt_state)

    if verbose:
      with jax.numpy.printoptions(precision=2, suppress=True):
        print((f"\nAt {num_update:5} "
               f"ELBO {float(ELBO_array[num_update]):8.3f}  "
               f"log-lik {float(log_likelihood):8.3f}  "
               f"entropy {float(entropy):8.3f}"))

  q_param = np.array(q_func(q_logit))
  return q_param, ELBO_array, logits_array


def compute_elbo_manually_unstruc_vi(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    q_enumerated: np.ndarray,
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
  """Computes ELBO for unstructured VI."""
  np.testing.assert_almost_equal(np.sum(q_enumerated), 1.0, decimal=5)

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  log_A_start = np.array(log_A_start_list)

  # Find all sequences
  seq_product = list(itertools.product(potential_sequences, repeat=num_users))
  num_sequences_product = len(seq_product)
  print(f"Work with sequence product of length {num_sequences_product}")

  log_joints = np.zeros((num_sequences_product))

  for num_sample, samples in enumerate(seq_product):
    samples_current = {u: samples[u] for u in range(num_users)}
    log_joints[num_sample] = util.calc_log_joint(
      samples=samples_current,
      observations=observations_all,
      contacts=contacts_all,
      prior=(potential_sequences, log_A_start),
      num_users=num_users,
      num_time_steps=num_time_steps,
      alpha=alpha,
      beta=beta,
      probab_0=probab_0,
      probab_channels=np.array([probab_1]))

  log_likelihood = q_enumerated.dot(log_joints)
  entropy = -q_enumerated.dot(jnp.log(q_enumerated))

  return log_likelihood, entropy


def struc_var_inf_jax(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01,
    num_updates: int = 1000,
    num_inner_steps: int = 5,
    learning_rate: float = 0.5,
    momentum: float = 0.5,
    verbose: bool = False):
  """Structural Variational Inference to infer latent SEIR states using JAX."""

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  q_marginal_infected = np.zeros((num_users, num_time_steps))

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  def learning_rate_func(step):
    mult_exp = int(2.2 * step/num_updates)
    mult = 2**(-mult_exp)
    return learning_rate * mult

  optimizer_list = []
  num_params = 1+3*num_time_steps
  for _ in range(num_users):
    opt_init, opt_update, opt_get = optimizers.nesterov(
      learning_rate_func, momentum)
    opt_state = opt_init(jnp.array(1+np.random.randn(num_params)))
    optimizer_list.append((opt_state, opt_update, opt_get))

  def log_q_func(logits):
    params = jax.nn.sigmoid(logits)
    return util.enumerate_log_q_values_untied_unrolled_beta_jax(
      params, seq_array, num_time_steps)

  grad_log_q_func = jax.jacfwd(log_q_func)

  ELBO = None
  ELBO_array = np.zeros((num_updates))
  ELBO_inner = np.zeros((num_updates, num_users, num_inner_steps))
  param_array = np.zeros((num_updates, num_users, num_params))

  for num_update in range(num_updates):
    # Collect gradients over all users.
    # Assumes that no contacts exist between users

    ELBO = np.array([0.])
    for user in range(num_users):

      d_term, d_no_term = util.precompute_d_penalty_terms_vi(
        q_marginal_infected,
        p0=probab_0,
        p1=probab_1,
        past_contacts=infect_counter.get_past_contacts(user),
        num_time_steps=num_time_steps)
      d_noterm_cumsum = np.cumsum(d_no_term)

      d_penalties = (
        np.take(d_noterm_cumsum, np.maximum(seq_array[:, 0]-1, 0))
        + np.take(d_term, seq_array[:, 0]))

      # Obtain q
      opt_state, opt_update, opt_get = optimizer_list[user]
      for num_inner_step in range(num_inner_steps):
        q_logit_u = opt_get(opt_state)
        log_q_vec = log_q_func(q_logit_u)
        q_vec = jnp.exp(log_q_vec)
        grad_q_vec = grad_log_q_func(q_logit_u).T * q_vec

        log_joint = log_c_z_u[user] + log_A_start + d_penalties
        # if num_update > 100:
        #   import pdb; pdb.set_trace()

        # Multiply with
        likelihood = q_vec.dot(log_joint)
        entropy = -q_vec.dot(log_q_vec)
        ELBO_user = likelihood + entropy
        ELBO_inner[num_update, user, num_inner_step] = ELBO_user

        likelihood_grads = grad_q_vec.dot(log_joint)
        entropy_grads = grad_q_vec.dot(-(1 + log_q_vec))
        # Multiply with -1 for maximisation
        grad = -1*(likelihood_grads + entropy_grads)
        opt_state = opt_update(num_inner_step, grad, opt_state)
      ELBO += ELBO_user
      optimizer_list[user] = (opt_state, opt_update, opt_get)

      # Update q_marginal_infected
      q_param_u = jax.nn.sigmoid(opt_get(opt_state))
      q_marginal_infected[user] = util.get_q_marginal_infected(
        q_param_u, num_time_steps)
      param_array[num_update, user] = np.array(q_param_u)

    if jnp.any(jnp.isinf(grad)) or jnp.any(jnp.isnan(grad)):
      import pdb; pdb.set_trace()  # pylint: disable=import-outside-toplevel,multiple-statements,forgotten-debug-statement

    ELBO_array[num_update] = float(ELBO)

    if verbose:
      with jax.numpy.printoptions(precision=2, suppress=True):
        print((f"At {num_update:5} "
               f"New marginal infected user 0: {q_marginal_infected[0]} "
               f"at ELBO {ELBO}"))

  return ELBO_array, param_array, ELBO_inner


def var_inf_unparam(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01,
    num_updates: int = 1000,
    verbose: bool = False,
    diagnostic: Optional[Any] = None):
  """Variational Inference without parameters."""
  del diagnostic

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  ELBO_array = np.zeros((num_updates))
  post_exps = np.zeros((num_users, num_time_steps, 4))
  for num_update in range(num_updates):
    # Collect gradients over all users.
    # Assumes that no contacts exist between users

    ELBO = 0.
    for user in range(num_users):

      d_term, d_no_term = util.precompute_d_penalty_terms_vi(
        post_exps[:, :, 2],
        p0=probab_0,
        p1=probab_1,
        past_contacts=infect_counter.get_past_contacts(user),
        num_time_steps=num_time_steps)
      d_noterm_cumsum = np.cumsum(d_no_term)

      d_penalties = (
        np.take(d_noterm_cumsum, np.maximum(seq_array[:, 0]-1, 0))
        + np.take(d_term, seq_array[:, 0]))

      log_joint = log_c_z_u[user] + log_A_start + d_penalties

      log_q_states = log_joint - special.logsumexp(log_joint)
      q_states = np.exp(log_q_states)

      ELBO += q_states.dot(log_joint) - q_states.dot(log_q_states)

      # Update posterior expectations
      post_exps[user] = seq_array_hot.dot(np.exp(log_q_states))

    if verbose:
      with jax.numpy.printoptions(precision=2, suppress=True):
        print((f"At {num_update:5} "
               f"New marginal infected user 0: {post_exps[0, :, 2]} "
               f"at ELBO {ELBO}"))
    ELBO_array[num_update] = ELBO

  return post_exps, ELBO_array


def compute_elbo_manually_struc_vi(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    q_params: Dict[int, Any],
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01):
  """Computes ELBO manually for a parameter setting."""

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  q_marginal_infected = np.zeros((num_users, num_time_steps))
  # Update q_marginal_infected
  for user in range(num_users):
    q_marginal_infected[user] = util.get_q_marginal_infected(
      q_params[user], num_time_steps)

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  def log_q_func(params):
    return util.enumerate_log_q_values_untied_unrolled_beta_jax(
      params, seq_array, num_time_steps)

  ELBO = 0.0

  for user in range(num_users):

    d_term, d_no_term = util.precompute_d_penalty_terms_vi(
      q_marginal_infected,
      p0=probab_0,
      p1=probab_1,
      past_contacts=infect_counter.get_past_contacts(user),
      num_time_steps=num_time_steps)
    d_noterm_cumsum = np.cumsum(d_no_term)

    d_penalties = (
      np.take(d_noterm_cumsum, np.maximum(seq_array[:, 0]-1, 0))
      + np.take(d_term, seq_array[:, 0]))

    # Obtain q
    log_q_vec = log_q_func(q_params[user])
    q_vec = jnp.exp(log_q_vec)

    log_joint = log_c_z_u[user] + log_A_start + d_penalties
    # if num_update > 100:
    #   import pdb; pdb.set_trace()

    # Multiply with
    likelihood = q_vec.dot(log_joint)
    entropy = -q_vec.dot(log_q_vec)
    ELBO += likelihood + entropy

  return ELBO


def struc_var_inf_jax_single(  # pylint: disable=dangerous-default-value
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    num_time_steps: int,
    observations_all: List[constants.Observation],
    contacts_all: List[constants.Contact],
    probab_0: float,
    probab_1: float,
    alpha: float = 0.001,
    beta: float = 0.01,
    num_updates: int = 1000,
    learning_rate: float = 0.5,
    momentum: float = 0.5,
    verbose: bool = False):
  """Structural Variational Inference to infer latent SEIR states using JAX."""
  if len(contacts_all) > 0:
    print("Warning: only implemented for no contacts")
  del momentum

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  q_marginal_infected = np.zeros((num_users, num_time_steps))

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  def learning_rate_func(step):
    mult_exp = int(3 * step/num_updates)
    mult = 10**(-mult_exp)
    return learning_rate * mult

  num_params = 1+3*num_time_steps
  opt_init, opt_update, opt_get = optimizers.adam(
    learning_rate_func)
  opt_state = opt_init(
    {0: jnp.array(1+np.random.randn(num_params))})
  q_logit = opt_get(opt_state)

  q_logit_1 = jnp.array([
    1.4108326, 1.2508416, 1.2181981, 1.539925,
    6.8258843, 6.141754, 6.8868737, 0.5379833,
    1.4168769, 0.8759312, 0.1025261, -5.268521,
    1.0796521, 1.0188112, 0.9311580, 1.0991639,
    7.6879067, 6.243615, 6.2261553, 1.2293478,
    1.0676554, 0.97909844])

  def log_q_func(logits):
    params = jax.nn.sigmoid(logits)
    return util.enumerate_log_q_values_untied_unrolled_beta_jax(
      params, seq_array, num_time_steps)

  grad_log_q_func = jax.jacfwd(log_q_func)

  ELBO = None
  ELBO_array = np.zeros((num_updates))
  param_array = np.zeros((num_updates, num_users, num_params))

  for num_update in range(num_updates):
    # Collect gradients over all users.
    # Assumes that no contacts exist between users

    grads = {user: jnp.zeros((num_params)) for user in range(num_users)}
    ELBO = np.array([0.])
    for user in range(num_users):

      d_term, d_no_term = util.precompute_d_penalty_terms_vi(
        q_marginal_infected,
        p0=probab_0,
        p1=probab_1,
        past_contacts=infect_counter.get_past_contacts(user),
        num_time_steps=num_time_steps)
      d_noterm_cumsum = np.cumsum(d_no_term)

      d_penalties = (
        np.take(d_noterm_cumsum, np.maximum(seq_array[:, 0]-1, 0))
        + np.take(d_term, seq_array[:, 0]))

      # Obtain q
      q_logit_u = q_logit[0] if user == 0 else q_logit_1
      log_q_vec = log_q_func(q_logit_u)
      q_vec = jnp.exp(log_q_vec)
      grad_q_vec = grad_log_q_func(q_logit_u).T * q_vec

      log_joint = log_c_z_u[user] + log_A_start + d_penalties
      # if (num_update > 100):
      #   import pdb; pdb.set_trace()

      # Multiply with
      likelihood = q_vec.dot(log_joint)
      entropy = -q_vec.dot(log_q_vec)
      ELBO += likelihood + entropy

      likelihood_grads = grad_q_vec.dot(log_joint)
      entropy_grads = grad_q_vec.dot(-(1 + log_q_vec))
      grads[user] = -1*(likelihood_grads + entropy_grads)

    if jnp.any(jnp.isinf(grads[0])) or jnp.any(jnp.isnan(grads[0])):
      import pdb; pdb.set_trace()  # pylint: disable=import-outside-toplevel,multiple-statements,forgotten-debug-statement

    # Update q_marginal_infected for single user 0
    q_param_u = jax.nn.sigmoid(q_logit[0])
    q_marginal_infected[0] = util.get_q_marginal_infected(
      q_param_u, num_time_steps)
    q_marginal_infected[1] = util.get_q_marginal_infected(
      jax.nn.sigmoid(q_logit_1), num_time_steps)
    param_array[num_update, 0] = np.array(q_param_u)

    opt_state = opt_update(num_update, {0: grads[0]}, opt_state)

    q_logit = opt_get(opt_state)

    ELBO_array[num_update] = float(ELBO)

    if verbose:
      with jax.numpy.printoptions(precision=2, suppress=True):
        print((f"At {num_update:5} "
               f"New beta params for user 0: {util.sigmoid(q_logit[0])} "
               f"at ELBO {ELBO}"))

  q_param = q_logit
  return q_param, ELBO_array, param_array
