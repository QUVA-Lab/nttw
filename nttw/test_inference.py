"""Test functions for inference.py."""

from nttw import inference, test_util, util
import numpy as np


def test_factorised_neighbor_step():

  contacts_all = [
    (0, 1, 2, [1]),
    (1, 0, 2, [1]),
    (3, 2, 2, [1]),
    (2, 3, 2, [1]),
    (4, 5, 2, [1]),
    (5, 4, 2, [1]),
    ]
  observations_all = [
    (0, 2, 1)
  ]
  num_users = 6
  num_time_steps = 5

  p0, p1 = 0.01, 0.3

  it_prior = util.iter_prior(
    *test_util.gen_dummy_dynamics(p0, 1/3, 1/3, num_time_steps),
    time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)
  seq_array = np.stack(potential_sequences, axis=0)
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0])

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=0.001, beta=0.01)

  q_marginal_infected = np.array([
    [.8, .8, .8, .8, .8],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
  ])

  user_slice_input = np.array([0, 1, 2])
  past_contacts = infect_counter.get_past_contacts_slice(user_slice_input)

  log_c_z_u_s = np.stack([log_c_z_u[user] for user in user_slice_input], axis=0)

  obs_diff = np.max(log_c_z_u_s) - np.min(log_c_z_u_s)
  assert obs_diff > 1.0, f"Observation difference is too small {obs_diff}"

  user_slice, post_exps, _, _ = inference.fn_step_wrapped(
    user_slice=user_slice_input,
    seq_array_hot=seq_array_hot,
    log_c_z_u=log_c_z_u_s,
    log_A_start=np.array(log_A_start_list),
    p_infected_matrix=q_marginal_infected,
    num_time_steps=num_time_steps,
    probab0=p0,
    probab1=p1,
    past_contacts_array=past_contacts)

  np.testing.assert_array_almost_equal(user_slice, user_slice_input)
  np.testing.assert_array_almost_equal(
    post_exps.shape, [len(user_slice_input), num_time_steps, 4])


def test_fact_neigh_with_start_belief():

  contacts_all = [
    (0, 1, 2, [1]),
    ]
  observations_all = [
    (0, 2, 1)
  ]
  num_users = 2
  num_time_steps = 5

  p0, p1 = 0.01, 0.5

  start_belief = np.array(
    [[.1, .4, .5, .0],
     [.9, .1, .0, .0]])

  post_exp, _ = inference.fact_neigh(
    num_users=num_users,
    num_time_steps=num_time_steps,
    observations_all=observations_all,
    contacts_all=contacts_all,
    probab_0=p0,
    probab_1=p1,
    g_param=.5,
    h_param=.5,
    start_belief=start_belief,
    alpha=0.001,
    beta=0.01,
    num_updates=5,
    num_jobs=1)

  text = ("Note this is a stochastic test. And may fail one in a thousand times"
          "Please rerun a few times")
  # Start belief for u0 is high in E and I states, so after the contact between
  # u0 and u1 on day 2, then u1 should be in E state and I state after

  with np.printoptions(precision=3, suppress=True):
    assert post_exp[1][3][1] > .2, text + "\n" + repr(post_exp)
    assert post_exp[1][4][2] > .1, text + "\n" + repr(post_exp)
