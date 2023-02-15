"""Plotting functions related to research around virus-spread modelling."""

# pylint: disable=duplicate-code
from nttw import constants, util
import numpy as np
from typing import List, Union


def plot_var_inf(  # pylint: disable=dangerous-default-value
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
    q_logit_default: Union[List[float], np.ndarray],
    beta2_values: Union[List[float], np.ndarray],
    alpha: float = 0.001,
    beta: float = 0.01,
    ):
  """Generate data to plot ELBO for range of variational parameters."""
  if len(contacts_all) > 0:
    print("Warning: only implemented for no contacts")

  it_prior = util.iter_prior(
    log_q_s,
    log_q_s_tail,
    log_q_e,
    log_q_e_tail,
    log_q_i,
    log_q_i_tail,
    time_total=num_time_steps
    )

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)

  # Precompute log_c_z_u[user] and log_A terms. Defined after CRISP paper
  log_c_z_u = util.calc_c_z_u(seq_array, observations_all,
                              num_users=num_users, alpha=alpha, beta=beta)
  log_A_start = np.array(log_A_start_list)

  num_probes = len(beta2_values)

  ELBO_array = np.zeros((num_probes))

  for num_probe, beta2_value in enumerate(beta2_values):
    q_logit = np.copy(q_logit_default)
    q_logit[3] = beta2_value

    # Collect gradients over all users.
    # Assumes that no contacts exist between users
    q_param = util.sigmoid(q_logit)
    log_q_vec = util.enumerate_log_q_values(q_param, seq_array)
    q_vec = np.exp(log_q_vec)
    np.testing.assert_almost_equal(np.sum(q_vec), 1., decimal=4,
                                   err_msg=f'Found sum {np.sum(q_vec)}')

    ELBO = 0.
    for user in range(num_users):
      log_joint = log_c_z_u[user] + log_A_start

      likelihood = q_vec.dot(log_joint)
      entropy = -q_vec.dot(np.log(q_vec))
      ELBO += likelihood + entropy
    ELBO_array[num_probe] = ELBO

  return ELBO_array
