"""Utility functions for running experiments."""
import crisp  # pytype: disable=import-error
import itertools
import joblib
import numpy as np
from nttw import belief_propagation, inference, logger, scenarios, util
from nttw.experiments import util_sib
import os
import random
import sib  # pytype: disable=import-error
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def gibbs_multiprocess_func(
    num_gibbs: int,
    n_job: Any,
    num_users: int,
    num_time_steps: int,
    contacts_subset: List[Any],
    observations_subset: List[Any],
    pmf_e: np.ndarray,
    pmf_i: np.ndarray,
    alpha: float,
    beta: float,
    probab_0: float,
    probab_1: float) -> Tuple[Any, np.ndarray]:
  """Inner function for the Gibbs sampler, to be used in multiprocessing."""

  qE = crisp.Distribution(pmf_e.tolist())
  qI = crisp.Distribution(pmf_i.tolist())

  num_burnin = min((num_gibbs, 10))
  result = crisp.GibbsPIS(
    num_users, num_time_steps, contacts_subset, observations_subset, qE, qI,
    alpha, beta, probab_0, probab_1, False)
  return n_job, result.get_marginals(num_gibbs, burnin=num_burnin)


def wrap_gibbs_inference(
    num_users: int,
    pmf_e: np.ndarray,
    pmf_i: np.ndarray,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    num_buckets: int,):
  """Wraps the inference function runs Gibbs sampling."""
  num_jobs = max((num_buckets, 1))  # Default value is -1

  def gibbs_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, start_belief

    if users_stale is not None:
      raise ValueError('Not implemented stale users for Gibbs')

    observations_list = scenarios.make_plain_observations(observations_list)
    contacts_list = scenarios.make_plain_contacts(contacts_list)

    post_exps = np.zeros((num_users, num_time_steps, 4))

    # Note: this will be slow for lots of chains
    with joblib.Parallel(n_jobs=num_jobs, backend="loky") as parallel:
      results = parallel(joblib.delayed(gibbs_multiprocess_func)(
        num_updates, n_job, num_users, num_time_steps, contacts_list,
        observations_list, pmf_e, pmf_i, alpha, beta, p0, p1
      ) for n_job in range(num_jobs))

      for (_, post_exp) in results:
        post_exps += post_exp

    return post_exps / num_jobs

  return gibbs_wrapped


def wrap_pgibbs_inference(
    num_users: int,
    g_param: float,
    h_param: float,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    num_buckets: int,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs Factorised Neighbors"""
  num_jobs = max((num_buckets, 1))  # Default value is -1

  def pgibbs_multiprocess_func(
      n_job: Any,
      observations_list: List[Any],
      contacts_list: List[Any],
      num_samples: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):

    # Set seed using n_job
    seed = random.randrange(100) + n_job
    random.seed(seed)
    np.random.seed(seed)

    num_burn = min((num_samples, 10))
    traces_in_triplets = inference.gibbs_sampling(
      num_users=num_users,
      g_param=g_param,
      h_param=h_param,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_channels=np.array([p1]),
      probab_0=p0,
      start_belief=start_belief,
      num_samples=num_samples,
      skip_steps=1,
      num_burn=num_burn,
      verbose=False,
      diagnostic=diagnostic)

    # traces_in_triplets is a matrix in [num_users, num_updates, 3], where the
    # last dimension are triplets of (t_s, t_e, t_i). Next line converts these
    # to the trace of SEIR states.
    traces_in_SEIR = np.reshape(util.state_seq_to_hot_time_seq(
      np.reshape(traces_in_triplets, [-1, 3]).astype(np.int32),
      time_total=num_time_steps
    ), [num_users, num_samples, num_time_steps, 4])

    if trace_dir:
      fname = os.path.join(trace_dir, "trace_pgibbs.npy")
      with open(fname, 'wb') as fp:
        np.save(fp, traces_in_SEIR.astype(np.uint8))

    return n_job, np.mean(traces_in_SEIR, axis=1)

  def pgibbs_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):
    del start_belief
    if users_stale is not None:
      raise ValueError('Not implemented stale users for pGibbs')

    post_exps = np.zeros((num_users, num_time_steps, 4))

    # Note: this will be slow for lots of chains
    with joblib.Parallel(n_jobs=num_jobs, backend="loky") as parallel:
      results = parallel(joblib.delayed(pgibbs_multiprocess_func)(
        n_job, observations_list, contacts_list, num_updates, num_time_steps,
        diagnostic
      ) for n_job in range(num_jobs))

      for (_, post_exp) in results:
        post_exps += post_exp

    return post_exps / num_jobs

  return pgibbs_wrapped


def wrap_fact_neigh_inference(
    num_users: int,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    g_param: float,
    h_param: float,
    damping: float,
    quantization: int = -1,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function that runs Factorised Neighbors"""

  def fact_neigh_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):

    traces_per_user_fn, _ = inference.fact_neigh(
      num_users=num_users,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=p0,
      probab_1=p1,
      g_param=g_param,
      h_param=h_param,
      start_belief=start_belief,
      damping=damping,
      quantization=quantization,
      users_stale=users_stale,
      num_updates=num_updates,
      num_jobs=min((num_users, util.get_cpu_count())),
      trace_dir=trace_dir,
      diagnostic=diagnostic)
    return traces_per_user_fn
  return fact_neigh_wrapped


def wrap_variational_inference(
    num_users: int,
    log_q_s: np.ndarray,
    log_q_s_tail: np.ndarray,
    log_q_e: np.ndarray,
    log_q_e_tail: np.ndarray,
    log_q_i: np.ndarray,
    log_q_i_tail: np.ndarray,
    alpha: float,
    beta: float,
    p0: float,
    p1: float):
  """Wraps the inference function that runs Variational Inference"""
  def var_inf_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):
    del start_belief, users_stale

    traces_per_user_fn, _ = inference.var_inf_unparam(
      num_users=num_users,
      log_q_s=np.array(log_q_s),
      log_q_s_tail=np.array(log_q_s_tail),
      log_q_e=log_q_e,
      log_q_e_tail=log_q_e_tail,
      log_q_i=log_q_i,
      log_q_i_tail=log_q_i_tail,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=p0,
      probab_1=p1,
      num_updates=num_updates,
      diagnostic=diagnostic)
    return traces_per_user_fn
  return var_inf_wrapped


def wrap_belief_propagation(
    num_users: int,
    param_g: float,
    param_h: float,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    damping: float = 0.0,
    quantization: int = -1,
    freeze_backwards: bool = False,
    trace_dir: Optional[str] = None,
    ):
  """Wraps the inference function runs Belief Propagation."""

  A_matrix = np.array([
    [1-p0, p0, 0, 0],
    [0, 1-param_g, param_g, 0],
    [0, 0, 1-param_h, param_h],
    [0, 0, 0, 1]
  ])

  obs_distro = {
    0: np.array([1-beta, 1-beta, alpha, 1-beta]),
    1: np.array([beta, beta, 1-alpha, beta]),
  }

  def bp_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):

    # Contacts on last day are not of influence
    def filter_fn(contact):
      return (contact[2] + 1) < num_time_steps
    contacts_list = list(filter(filter_fn, contacts_list))

    obs_messages = np.ones((num_users, num_time_steps, 4))
    for obs in observations_list:
      if obs[1] < num_time_steps:
        obs_messages[obs[0]][obs[1]] *= obs_distro[obs[2]]

    map_forward_message, map_backward_message = (
      belief_propagation.init_message_maps(contacts_list, num_users))

    num_jobs = util.get_cpu_count()
    logger.info(f"Multiprocess BP with {num_jobs} jobs")
    with joblib.Parallel(n_jobs=num_jobs, backend="loky") as parallel:
      for num_update in range(num_updates):
        # logger.info(f"Start round {num_update}")

        damping_use = damping
        if num_update == 0:
          damping_use = 0.0

        (bp_beliefs, map_backward_message, map_forward_message) = (
          belief_propagation.do_backward_forward_and_message(
            A_matrix, p0, p1, num_time_steps, obs_messages, num_users,
            map_backward_message, map_forward_message,
            damping=damping_use,
            start_belief=start_belief,
            quantization=quantization,
            freeze_backwards=freeze_backwards,
            users_stale=users_stale,
            parallel=parallel,
            num_jobs=num_jobs))

        if trace_dir:
          fname = os.path.join(trace_dir, f"trace_{num_update:05d}.npy")
          with open(fname, 'wb') as fp:
            np.save(fp, bp_beliefs)

        if diagnostic:
          diagnostic.log({f'debug_value': -1.}, commit=False)

    bp_beliefs /= np.sum(bp_beliefs, axis=-1, keepdims=True)
    return bp_beliefs
  return bp_wrapped


def wrap_sib(
    num_users: int,
    recovery_days: float,
    p0: float,
    p1: float,
    damping: float):
  """Wraps the inference function for the SIB library.

  https://github.com/sibyl-team/sib"""

  def sib_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None):
    del start_belief

    if users_stale is not None:
      raise ValueError('Not implemented stale users for Gibbs')

    # Prepare contacts
    contacts_plain = scenarios.make_plain_contacts(contacts_list)
    contacts_sib = []
    for contact in contacts_plain:
      contact = list(contact)
      contact[3] = p1
      contacts_sib.append(tuple(contact))

    # Prepare observations
    obs_sib = []
    for obs in observations_list:
      if obs['outcome'] == 1:
        test = sib.Test(0, 1, 0)
      else:
        test = sib.Test(.5, 0, .5)
      obs_sib.append((obs[0], test, obs[1]))
    # Add dummy observations to query marginals later
    obs_sib += [(i, sib.Test(1, 1, 1), timestep)
                for i in range(num_users) for timestep in range(num_time_steps)]

    # Sort observations and contacts (required for SIB library)
    obs_sib = list(sorted(obs_sib, key=lambda x: x[2]))
    contacts_sib = list(sorted(contacts_sib, key=lambda x: x[2]))

    sib_params = util_sib.make_sib_params(num_time_steps, p0, recovery_days)

    # Run inference
    f = sib.FactorGraph(
      params=sib_params,
      contacts=contacts_sib,
      tests=obs_sib)

    if diagnostic:
      diagnostic.log({"damping": damping}, commit=False)
    sib.iterate(f, maxit=num_updates, damping=damping)

    nodes_all = {node.index: node for node in f.nodes}

    # Collect marginals
    marginals_sib = np.zeros((num_users, num_time_steps, 3))
    for user in range(num_users):
      for timestep in range(num_time_steps):
        message = sib.marginal_t(nodes_all[user], timestep)
        marginals_sib[user, timestep] = np.array(list(message))

    logger.info(f"Marginals SIB contain NaN {np.any(np.isnan(marginals_sib))}")

    # Insert a slice with all zeros for E state (SIB does only SIR)
    marginals_sib = np.concatenate(
      (marginals_sib[:, :, :1],
       np.zeros((num_users, num_time_steps, 1)),
       marginals_sib[:, :, 1:]),
      axis=-1)
    return marginals_sib
  return sib_wrapped


def wrap_dummy_inference(
    num_users: int,):
  """Wraps the inference function for dummy inference."""

  def dummy_wrapped(
      observations_list: List[Any],
      contacts_list: List[Any],
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, start_belief, num_updates, contacts_list, observations_list
    del users_stale

    predictions = np.random.randn(num_users, num_time_steps, 4)
    predictions /= np.sum(predictions, axis=-1, keepdims=True)

    return predictions

  return dummy_wrapped


def set_noisy_test_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
  """Sets the noise parameters of the observational model."""
  noise_level = cfg["model"]["noisy_test"]
  assert 0 <= noise_level <= 3

  if noise_level == 0:
    return cfg

  alpha_betas = [(), (.01, .001), (.1, .01), (.25, .03)]

  # Set model parameters
  cfg["model"]["alpha"] = alpha_betas[noise_level][0]
  cfg["model"]["beta"] = alpha_betas[noise_level][1]

  # Don't assume model misspecification, set data parameters the same
  cfg["data"]["alpha"] = alpha_betas[noise_level][0]
  cfg["data"]["beta"] = alpha_betas[noise_level][1]

  return cfg


def make_git_log():
  """Logs the git diff and git show.

  Note that this function has a general try/except clause and will except most
  errors produced by the git commands.
  """
  try:
    result = subprocess.run(
      ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git show \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git diff \n{result.stdout.decode('utf-8')}")
  except Exception as e:  # pylint: disable=broad-except
    logger.info(f"Git log not printed due to {e}")
