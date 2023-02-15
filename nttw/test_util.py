"""Test functions for util.py."""

import functools
import itertools
import jax  # pylint: disable=unused-import
import jax.numpy as jnp
from nttw import util
import numpy as np
import random
from scipy import special


def test_state_time():
  result = util.state_at_time([6, 0, 0], 4)
  assert float(result.flatten()[0]) == 0, f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 4)
  assert result.tolist() == [1, 2, 2], f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 1)
  assert result.tolist() == [1, 1, 1], f"result {result}"

  result = util.state_at_time([1, 1, 1], 4)
  assert float(result.flatten()[0]) == 3, f"result {result}"


def test_state_time_cache():
  result = util.state_at_time_cache(6, 0, 0, 4)
  assert result == 0

  result = util.state_at_time_cache(1, 4, 5, 4)
  assert result == 1

  result = util.state_at_time_cache(1, 3, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 1)
  assert result == 1

  result = util.state_at_time_cache(1, 2, 5, 8)
  assert result == 3


def test_precompute_p_u_t():
  samples_current = {
    # Being: t0, de, di ('time in state 0', 'days excited' and 'days infected')
    1: [1, 1, 4],
    2: [6, 0, 0],
  }
  probab_channels = np.array([.9])
  contacts_all = [
    (1, 2, 4, [1]),
    (2, 1, 4, [1])]

  counter = util.InfectiousContactCount(
    contacts_all, samples_current, num_users=6, num_time_steps=7)

  # First test when other user (v=1) is infected
  log_p_u_t = util.precompute_p_u_t(
    time_total=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels
    )
  expected = np.log(np.array([1., 1., 1., 1., 0.1, 1., 1.]))

  np.testing.assert_array_almost_equal(log_p_u_t, expected)

  # Second test when other user (v=1) is not infected, but susceptible
  samples_current[1] = [6, 0, 0]
  log_p_u_t = util.precompute_p_u_t(
    time_total=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels
    )
  expected = np.log(np.array([1., 1., 1., 1., 1., 1., 1.]))
  np.testing.assert_array_almost_equal(log_p_u_t, expected)

  # Third test when other user (v=1) is already immune
  samples_current[1] = [1, 1, 1]
  log_p_u_t = util.precompute_p_u_t(
    time_total=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels
    )
  expected = np.log(np.array([1., 1., 1., 1., 1., 1., 1.]))
  np.testing.assert_array_almost_equal(log_p_u_t, expected)

  # Fourth test when multiple contacts happen
  samples_current[3] = [1, 1, 4]
  samples_current[4] = [1, 2, 3]
  samples_current[5] = [1, 2, 3]
  contacts_all += [
    (3, 2, 4, [1]),
    (2, 3, 4, [1]),
    (4, 2, 4, [1]),
    (2, 4, 4, [1]),
    (5, 2, 4, [1]),
    (2, 5, 4, [1])]

  # Reconstruct counter, as InfectiousCounter does not handle changing contacts
  counter = util.InfectiousContactCount(
    contacts_all, samples_current, num_users=6, num_time_steps=7)
  log_p_u_t = util.precompute_p_u_t(
    time_total=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels
    )
  expected = np.log(np.array([1., 1., 1., 1., 0.001, 1., 1.]))
  np.testing.assert_array_almost_equal(log_p_u_t, expected)


def test_b_z_precomputation():
  samples_current = {
    # Being: t0, de, di ('time in state 0', 'days excited' and 'days infected')
    1: [5, 3, 3],
    2: [6, 0, 0],
    3: [11, 0, 0],
  }
  probab_channels = np.array([.8])
  probab_0 = 0.01  # Probability for exogenous infection at a given time step
  contacts_all = [
    (1, 2, 4, [1]),
    (2, 1, 4, [1]),
    (1, 3, 4, [1]),
    (3, 1, 4, [1])]
  counter = util.InfectiousContactCount(contacts_all, samples_current, 4, 11)

  log_b_ratio = util.precompute_b_z(
    total_time=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels,
    probab_0=probab_0
  )
  # 80.2 = (1 - (1-0.01)*(1-.8)**1) / (1 - (1-0.01)*(1-.8)**0)
  expected_b_ratio = np.log(np.array([1., 1., 1., 1., 80.2, 1., 1.]))
  np.testing.assert_array_almost_equal(log_b_ratio, expected_b_ratio)

  # ##########
  # Now repeat test when there's no jump from S to E
  counter.update_infect_count(1, samples_current[1], remove=True)
  samples_current[1] = [6, 0, 0]
  counter.update_infect_count(1, samples_current[1], remove=False)
  log_b_ratio = util.precompute_b_z(
    total_time=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels,
    probab_0=probab_0
  )

  # f = (1-p0) * (1-.8) / (1-p0)
  expected_b_ratio = np.log(np.array([1., 1., 1., 1., .2, 1., 1.]))
  np.testing.assert_array_almost_equal(log_b_ratio, expected_b_ratio)

  # ########
  # Now test when user 1 had multiple contacts
  counter.update_infect_count(1, samples_current[1], remove=True)
  samples_current[1] = [5, 3, 3]
  counter.update_infect_count(1, samples_current[1], remove=False)
  samples_current[3] = [1, 1, 4]
  counter.update_infect_count(3, samples_current[3], remove=False)

  log_b_ratio = util.precompute_b_z(
    total_time=6,
    user_u=2,
    infect_counter=counter,
    samples_current=samples_current,
    probab_channels=probab_channels,
    probab_0=probab_0
  )
  # 1.1975 = (1 - (1-0.01)*(1-.8)**2) / (1 - (1-0.01)*(1-.8)**1)
  expected_b_ratio = np.log(np.array([1., 1., 1., 1., 1.1975, 1., 1.]))
  np.testing.assert_array_almost_equal(log_b_ratio, expected_b_ratio, decimal=4)


def test_calculate_log_c_z():
  potential_sequences = list(util.generate_sequence_days(time_total=4))

  observations_all = [
    (1, 2, 1),
    (2, 3, 0),
    ]

  a, b = .1, .2

  result = util.calc_c_z_u(
    potential_sequences=np.array(potential_sequences),
    observations=observations_all,
    num_users=3,
    alpha=a,
    beta=b)
  expected = {
    0: [0, 0, 0, 0, 0, 0, 0, 0],
    1: [np.log(b), np.log(b), np.log(b), np.log(b), np.log(1-a), np.log(1-a),
        np.log(b), np.log(b)],
    2: [np.log(1-b), np.log(1-b), np.log(a), np.log(1-b), np.log(1-b),
        np.log(a), np.log(a), np.log(1-b)]}

  assert set(result) == set(expected), (f"Not all expected users found."
                                        f"found {set(result)}, "
                                        f"but expected {set(expected)}")

  for user, expected_c in expected.items():
    np.testing.assert_array_almost_equal(
      expected_c, result[user], err_msg=f'Error in {user}')


def test_generate_sequences():
  result = util.generate_sequence_days(time_total=4)
  expected = [(4, 0, 0), (3, 1, 0), (2, 1, 1), (2, 2, 0), (1, 1, 1), (1, 1, 2),
              (1, 2, 1), (1, 3, 0)]

  for x, y in zip(result, expected):
    assert x == y, f"{x} and {y} do not match"


def test_calc_log_a():
  potential_sequences = list(util.generate_sequence_days(time_total=4))
  seq_array = np.stack(potential_sequences, axis=0)

  g = 1 / 8
  h = 1 / 8
  p0 = 0.01

  log_A = util.calc_log_a_start(seq_array, p0, g, h)

  expected = [
    3*np.log(1-p0),
    2*np.log(1-p0),
    np.log(1-p0) + np.log(g),
    np.log(1-p0) + np.log(1-g),
    np.log(g) + np.log(h),
    np.log(g) + np.log(1-h),
    np.log(1-g) + np.log(g),
    2*np.log(1-g)
  ]
  np.testing.assert_array_almost_equal(log_A, expected)


def test_q_log_tail_prob():
  g = 1/8
  q_vec = [0] + [g*(1-g)**(i-1) for i in range(1, 1000)]
  pdf_e_normalised = [float(q)/sum(q_vec) for q in q_vec]

  log_pdf, log_pdf_tail = util.q_log_tail_prob(pdf_e_normalised)

  log_pdf_expected = np.array(
    [np.inf, -2.0794, -2.2130, -2.3465, -2.4800, -2.6136,
     -2.7471, -2.8806, -3.0142, -3.1477, -3.2812, -3.4148, -3.5483])

  assert np.isneginf(log_pdf[0])
  np.testing.assert_array_almost_equal(
    log_pdf[1:13], log_pdf_expected[1:13], decimal=3)

  log_pdf_tail_expected = np.array(
    [0.0000, 0.0000, -0.1335, -0.2671, -0.4006, -0.5341, -0.6677, -0.8012,
     -0.9347, -1.0683, -1.2018, -1.3353, -1.4688])
  np.testing.assert_array_almost_equal(
    log_pdf_tail[:13], log_pdf_tail_expected[:13], decimal=3)


def test_state_seq_to_time_seq():
  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_time_seq(seq_days, 5)

  expected = [
    [0, 0, 0, 0, 3],
    [0, 0, 0, 1, 3],
    [0, 0, 1, 2, 3],
    [0, 0, 1, 1, 3],
    [0, 1, 2, 3, 3],
    [0, 1, 2, 2, 3],
    [0, 1, 1, 2, 3],
    [0, 1, 1, 1, 3]]

  np.testing.assert_array_almost_equal(result, np.array(expected))

  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_hot_time_seq(seq_days, 5)
  np.testing.assert_array_almost_equal(result.shape, [8, 5, 4])


def gen_dummy_dynamics(p0, g, h, num_time_steps: int = 35, mult: int = 1):
  """Generate some dynamics for unit tests."""
  # g Probability to flip from E to I (on a given day)
  # h Probability to flip from I to R (on a given day)

  q_e_vec = [0] + [g*(1-g)**(i-1) for i in range(1, mult*num_time_steps+1)]
  q_i_vec = [0] + [h*(1-h)**(i-1) for i in range(1, mult*num_time_steps+1)]

  pdf_e = np.array(q_e_vec) / np.sum(q_e_vec)
  pdf_i = np.array(q_i_vec) / np.sum(q_i_vec)

  l_q_e, l_q_e_tail = util.q_log_tail_prob(pdf_e)
  l_q_i, l_q_i_tail = util.q_log_tail_prob(pdf_i)

  # import pdb; pdb.set_trace()

  l_q_s = [k*np.log(1 - p0) + np.log(p0) for k in range(mult*num_time_steps+1)]
  l_q_s_tail = [k * np.log(1 - p0) for k in range(mult*num_time_steps+1)]
  return l_q_s, l_q_s_tail, l_q_e, l_q_e_tail, l_q_i, l_q_i_tail


def test_iter_prior():
  seqs_expect = [
    (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 1), (0, 2, 2), (0, 2, 3),
    (0, 3, 1), (0, 3, 2), (0, 4, 1), (0, 5, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3),
    (1, 2, 1), (1, 2, 2), (1, 3, 1), (1, 4, 0), (2, 1, 1), (2, 1, 2), (2, 2, 1),
    (2, 3, 0), (3, 1, 1), (3, 2, 0), (4, 1, 0), (5, 0, 0)]
  logps_expect = [
    -7.29009, -7.42362, -7.55715, -7.06207, -7.44424, -7.57777, -6.74028,
    -7.59839, -6.53480, -6.39240, -6.54655, -7.30014, -7.43367, -6.59618,
    -7.45429, -6.39070, -6.24830, -5.78341, -7.31019, -6.24660, -6.10420,
    -5.30598, -5.96010, -4.94438, -4.64537, -0.05025
  ]

  num_time_steps = 5

  dynamics = gen_dummy_dynamics(0.01, 1/7, 1/8, num_time_steps)

  it_prior = util.iter_prior(
    *dynamics,
    time_total=num_time_steps
    )

  seqs, logps = zip(*it_prior)

  np.testing.assert_array_almost_equal(np.stack(seqs), np.stack(seqs_expect),
                                       decimal=5)
  np.testing.assert_array_almost_equal(np.stack(logps), np.stack(logps_expect),
                                       decimal=5)

  # Also test direct iteration of traces
  seqs_direct = list(util.iter_sequences(num_time_steps, start_se=True))
  np.testing.assert_array_almost_equal(np.stack(seqs), np.stack(seqs_direct))


def test_iter_sequences():
  num_time_steps = 7
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=True)))
  np.testing.assert_almost_equal(num_seqs, 64)

  # num_time_steps+1 more
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=False)))
  np.testing.assert_almost_equal(num_seqs, 72)


def test_enumerate_log_prior_values():
  num_time_steps = 7
  p0, g, h = 0.01, 0.2, 0.16

  seqs = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=True)))

  log_p = util.enumerate_log_prior_values(
    [1-p0, p0, 0., 0.], [1-p0, 1-g, 1-h], seqs, num_time_steps)

  np.testing.assert_almost_equal(np.sum(np.exp(log_p)), 1.0, decimal=3)

  # Test it_prior against NumPy approach
  it_prior = util.iter_prior(
    *gen_dummy_dynamics(p0, g, h, num_time_steps=num_time_steps, mult=100),
    time_total=num_time_steps)
  _, log_A_start_list = zip(*it_prior)
  log_A_start_list = np.array(list(log_A_start_list))

  np.testing.assert_array_almost_equal(log_A_start_list, log_p)


def test_enumerate_log_prior_values_full_sums_1():
  num_time_steps = 7
  p0, g, h = 0.01, 0.2, 0.16

  seqs = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))

  log_p = util.enumerate_log_prior_values(
    [1-p0, p0, 0., 0.], [1-p0, 1-g, 1-h], seqs, num_time_steps)

  np.testing.assert_almost_equal(np.sum(np.exp(log_p)), 1.0, decimal=3)


def test_log_joint_sums_1():
  alpha, beta, p0 = 0.001, 0.01, 0.001
  # Times (in days) for simulated data
  num_time_steps, num_users = 3, 1
  g = h = 1/3

  dynamics = gen_dummy_dynamics(p0, g, h, num_time_steps)
  it_prior = util.iter_prior(*dynamics, time_total=num_time_steps)
  potential_sequences, log_A_start_list = zip(*it_prior)

  seq_array = np.array(list(potential_sequences))
  log_A_start = np.array(log_A_start_list)

  # Positive observation
  observations_all = [(0, 2, 1)]
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_p_pos = special.logsumexp(log_A_start + log_c_z_u[0])

  # Negative observation
  observations_all = [(0, 2, 0)]
  log_c_z_u = util.calc_c_z_u(
    seq_array, observations_all, num_users=num_users, alpha=alpha, beta=beta)
  log_p_neg = special.logsumexp(log_A_start + log_c_z_u[0])

  # joint p(x,z) should sum to 1 over all x and z
  np.testing.assert_almost_equal(np.exp(log_p_pos) + np.exp(log_p_neg), 1.0)


def test_infect_contact_count():
  contacts_all = [
    (0, 1, 6, [1]),
    (1, 0, 6, [1]),
    (0, 2, 5, [1]),
    (2, 0, 5, [1]),
    (2, 3, 5, [1]),
    (3, 2, 5, [1]),
    (2, 4, 6, [1]),
    (4, 2, 6, [1]),
    (5, 0, 6, [1]),
    ]

  samples_current = {
    # Being: t0, de, di
    0: [1, 1, 9],
    1: [1, 1, 9],
    2: [1, 1, 9],
    3: [1, 1, 9],
    4: [1, 1, 9],
    5: [1, 1, 9],
    }

  counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=samples_current,
    num_users=6,
    num_time_steps=12)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 2, 1])

  counter.update_infect_count(3, [1, 1, 9], remove=True)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 1, 1])

  counter.update_infect_count(3, [1, 2, 9], remove=False)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 2, 1])

  expected_future_contacts = [(5, 2, [1]), (6, 1, [1])]
  result = list(counter.get_future_contacts(0))
  assert expected_future_contacts == result, "Future contacts don't match."

  expected_past_contacts = [(5, 2, [1]), (6, 1, [1]), (6, 5, [1])]
  result = list(counter.get_past_contacts(0))
  assert expected_past_contacts == result, "Past contacts don't match."


def test_gather_infected_precontacts():
  num_time_steps = 8
  contacts_all = [
    (0, 1, 6, [1]),
    (1, 0, 6, [1]),
    (0, 2, 5, [1]),
    (2, 0, 5, [1]),
    (2, 3, 5, [1]),
    (3, 2, 5, [1]),
    (2, 4, 6, [1]),
    (4, 2, 6, [1]),
    (5, 0, 6, [1]),
    ]

  samples_current = {
    # Being: t0, de, di
    0: [1, 1, 9],
    1: [1, 1, 9],
    2: [1, 1, 9],
    3: [1, 1, 9],
    4: [1, 1, 9],
    5: [1, 1, 9],
    }

  counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=samples_current,
    num_users=6,
    num_time_steps=num_time_steps)

  result = util.gather_infected_precontacts(
    num_time_steps=num_time_steps,
    samples_current=samples_current,
    past_contacts=counter.get_past_contacts(user=0))

  expected = np.array([0, 0, 0, 0, 0, 1, 2, 0])
  np.testing.assert_array_almost_equal(result, expected)

  # Change state of contact by 1 to time different time step
  samples_current[1] = [11, 0, 0]

  result = util.gather_infected_precontacts(
    num_time_steps=num_time_steps,
    samples_current=samples_current,
    past_contacts=counter.get_past_contacts(user=0))

  expected = np.array([0, 0, 0, 0, 0, 1, 1, 0])
  np.testing.assert_array_almost_equal(result, expected)


def test_calc_grad_p1():
  # Hard to write out numerical gradient, but we can test quantitative cases
  p0 = 0.01
  p1 = .3
  time_s = 4

  # Gradient should not be affected by infectious preparents after t_s
  x1 = util.calc_grad_p1(np.array([0, 0, 0, 2, 0, 0, 13]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  x2 = util.calc_grad_p1(np.array([0, 0, 0, 2, 0, 0, 2]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  np.testing.assert_almost_equal(x1, x2)

  # Less infectious preparents at t_s should results in larger gradient
  x1 = util.calc_grad_p1(np.array([0, 0, 0, 1, 0, 0, 13]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  x2 = util.calc_grad_p1(np.array([0, 0, 0, 2, 0, 0, 2]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  assert x1 > x2

  # More infectious preparents before t_s should results in larger neg gradient
  x1 = util.calc_grad_p1(np.array([0, 1, 1, 2, 0, 0, 13]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  x2 = util.calc_grad_p1(np.array([0, 1, 5, 2, 0, 0, 2]),
                         probab_0=p0, probab_1=p1, time_s=time_s)
  assert x1 > x2


def test_calc_log_joint_evidence():
  num_time_steps = 5

  p0 = 0.001
  p1 = 0.9
  # Times (in days) for simulated data
  t_test = 4
  t_touch01 = 3
  observations_all = [
    (1, t_test, 1)]
  contacts_all = [
    (0, 1, t_touch01, [1]),
    (1, 0, t_touch01, [1]),
    ]
  num_users = 2

  dynamics = util.gen_dynamics_array(
    p0, 1/2, 1/2, num_time_steps=num_time_steps)

  it_prior = util.iter_prior(*dynamics, time_total=num_time_steps)

  potential_sequences, log_A_start_list = zip(*it_prior)

  seq_product = list(itertools.product(potential_sequences, repeat=num_users))
  num_sequences_product = len(seq_product)

  log_p_xz = np.zeros((num_sequences_product))

  for num_iter, samples in enumerate(seq_product):
    samples_current = {u: samples[u] for u in range(num_users)}

    log_p_xz[num_iter] = util.calc_log_joint(
      samples=samples_current,
      observations=observations_all,
      contacts=contacts_all,
      prior=(potential_sequences, log_A_start_list),
      num_users=num_users,
      num_time_steps=num_time_steps,
      alpha=0.01,
      beta=0.001,
      probab_0=p0,
      probab_channels=np.array([p1]),
      )

  evidence_manual = special.logsumexp(log_p_xz)
  np.testing.assert_almost_equal(evidence_manual, -5.95, decimal=1)


def test_calc_log_joint():
  num_time_steps = 11
  num_users = 6

  # First write scenario with two infectious contacts with transmissin (thus
  # evidence for large p1)
  contacts_all = [
    (0, 1, 3, [1]),
    (1, 0, 3, [1]),
    (0, 2, 3, [1]),
    (2, 0, 3, [1]),
    (2, 3, 3, [1]),
    (3, 2, 3, [1]),
    (2, 4, 3, [1]),
    (4, 2, 3, [1]),
    ]

  samples_current = {
    # Being: t0, de, di
    0: [1, 1, 9],
    1: [4, 3, 3],
    2: [4, 3, 3],
    3: [8, 1, 1],
    4: [8, 1, 1],
    5: [8, 1, 1],
    }

  observations_all = [
    (1, 4, 1),
    (2, 4, 1),
    (5, 4, 0),
    ]

  p0 = 0.01
  dynamics = gen_dummy_dynamics(p0, 1/3, 1/3, num_time_steps)

  it_prior = util.iter_prior(
    *dynamics,
    time_total=num_time_steps
    )
  potential_seqs, a_values = zip(*it_prior)

  result_normal = util.calc_log_joint(
    samples_current,
    observations_all,
    contacts_all,
    prior=(potential_seqs, a_values),
    num_users=num_users,
    num_time_steps=num_time_steps,
    alpha=0.01,
    beta=0.001,
    probab_0=p0,
    probab_channels=np.array([.8]))

  result_too_low = util.calc_log_joint(
    samples_current,
    observations_all,
    contacts_all,
    prior=(potential_seqs, a_values),
    num_users=num_users,
    num_time_steps=num_time_steps,
    alpha=0.01,
    beta=0.001,
    probab_0=p0,
    probab_channels=np.array([.01]))
  assert result_normal > result_too_low

  # Change to scenario with infectious contact without transmission (thus
  # evidence for low p1)
  samples_current[0] = [1, 1, 9]
  samples_current[1] = [8, 1, 1]
  samples_current[2] = [8, 1, 1]

  observations_all = [
    (0, 2, 1),
    (1, 4, 0),
    (2, 4, 0),
    (5, 4, 0),
    ]

  result_too_high = util.calc_log_joint(
    samples_current,
    observations_all,
    contacts_all,
    prior=(potential_seqs, a_values),
    num_users=num_users,
    num_time_steps=num_time_steps,
    alpha=0.01,
    beta=0.001,
    probab_0=p0,
    probab_channels=np.array([.8]))

  result_normal = util.calc_log_joint(
    samples_current,
    observations_all,
    contacts_all,
    prior=(potential_seqs, a_values),
    num_users=num_users,
    num_time_steps=num_time_steps,
    alpha=0.01,
    beta=0.001,
    probab_0=p0,
    probab_channels=np.array([.01]))
  assert result_normal > result_too_high


def test_enumerate_q_grad_normalised():
  num_time_steps = 6

  dynamics = gen_dummy_dynamics(1/2, 2/3, 1/5, num_time_steps=num_time_steps)

  it_prior = util.iter_prior(*dynamics, num_time_steps)
  potential_sequences, log_A = zip(*it_prior)
  sequences = np.stack(potential_sequences, axis=0)

  log_q = util.enumerate_log_q_values_normalised(
    sequences=sequences,
    # Params are 1 minus params to generate dynamics.
    params=np.array([1/2, 1/3, 4/5]))

  log_A = np.array(log_A)
  log_q = np.array(log_q)

  np.testing.assert_array_almost_equal(log_A, log_q, decimal=5)
  np.testing.assert_almost_equal(np.sum(np.exp(log_q)), 1., decimal=5)
  np.testing.assert_almost_equal(np.sum(np.exp(log_A)), 1., decimal=5)


def test_enumerate_q():
  num_time_steps = 6

  dynamics = gen_dummy_dynamics(1/2, 2/3, 1/5, num_time_steps=num_time_steps)

  it_prior = util.iter_prior(*dynamics, num_time_steps)
  potential_sequences, _ = zip(*it_prior)
  sequences = np.stack(potential_sequences, axis=0)

  params = np.array([1/2, 7/9, 4/5, 2/7])

  log_q = util.enumerate_log_q_values(
    sequences=sequences,
    params=params)

  np.testing.assert_almost_equal(np.sum(np.exp(log_q)), 1., decimal=5)

  log_q = np.array(util.enumerate_log_q_values_jax(
    sequences=sequences,
    params=params,
    time_total=num_time_steps))

  np.testing.assert_almost_equal(np.sum(np.exp(log_q)), 1., decimal=5)

  params = np.array([1/2] + [2/7]*(3*num_time_steps))

  log_q = np.array(util.enumerate_log_q_values_untied_beta_jax(
    sequences=sequences,
    params=params,
    time_total=num_time_steps))

  np.testing.assert_almost_equal(np.sum(np.exp(log_q)), 1., decimal=5)


def test_autodiff_log_q():
  num_time_steps = 6

  dynamics = gen_dummy_dynamics(1/2, 2/3, 1/5, num_time_steps=num_time_steps)

  it_prior = util.iter_prior(*dynamics, num_time_steps)
  potential_sequences, _ = zip(*it_prior)
  sequences = np.stack(potential_sequences, axis=0)

  log_q_func = functools.partial(
    util.enumerate_log_q_values_jax, sequences=sequences,
    time_total=num_time_steps)
  grad_log_q_func = jax.jacfwd(log_q_func)

  grad_log_q = grad_log_q_func(jnp.array([1/2, 1/3, 1/3, 1/3]))
  np.testing.assert_array_almost_equal(grad_log_q.shape, [len(sequences), 4])

  # Also check the untied unrolling case
  log_q_func = functools.partial(
    util.enumerate_log_q_values_untied_unrolled_beta_jax, sequences=sequences,
    time_total=num_time_steps)
  grad_log_q_func = jax.jacfwd(log_q_func)

  logits = np.random.randn(3*num_time_steps+1)
  params = jax.nn.sigmoid(jnp.array(logits))

  log_q = log_q_func(params)
  np.testing.assert_almost_equal(
    np.sum(np.exp(np.array(log_q))), 1.0, decimal=5)

  grad_log_q = grad_log_q_func(params)
  np.testing.assert_array_almost_equal(grad_log_q.shape,
                                       [len(sequences), len(params)])


def test_autodiff_log_q_manual():
  num_time_steps = 6

  dynamics = gen_dummy_dynamics(1/2, 2/3, 1/5, num_time_steps=num_time_steps)

  it_prior = util.iter_prior(*dynamics, num_time_steps)
  potential_sequences, _ = zip(*it_prior)
  sequences = np.stack(potential_sequences, axis=0)

  log_q_func = functools.partial(
    util.enumerate_log_q_values_jax, sequences=sequences,
    time_total=num_time_steps)
  grad_log_q_func = jax.jacfwd(log_q_func)

  q_param = jnp.array([1/2, 1/3, 1/3, 1/3])

  grad_log_q = grad_log_q_func(q_param)

  grad_log_q_manual = util.enumerate_log_q_values_grad(q_param, sequences).T

  np.testing.assert_array_almost_equal(grad_log_q, grad_log_q_manual)


def test_estimate_h():
  samples = {
    # Being: t0, de, di
    0: [35, 0, 0],
    1: [15, 5, 5],
    2: [15, 5, 7],
    3: [25, 5, 5],
    4: [35, 0, 0],
    5: [35, 0, 0],
    }

  h_new = util.estimate_h(samples, num_users=6, num_time_steps=35)

  np.testing.assert_almost_equal(h_new, 2 / 16)


def test_sample_q():
  # NOTE: these are stochastic tests, they could fail by chance, but those
  # chances are VERY low.

  num_samples = 20

  q_param = np.array([0.99, 0.9, 0.9, 0.9])
  samples = util.sample_q(q_param, num_samples=num_samples, num_time=15,
                          collapse=False)
  num_start_0 = np.sum(samples[:, 0] <= 0)
  assert num_start_0 > (num_samples//2), (
    f"num_start_0 should be almost num_samples ({num_samples}), but is only"
    f"{num_start_0}")

  q_param = np.array([0.99, 0.1, 0.1, 0.1])
  samples = util.sample_q(q_param, num_samples=num_samples, num_time=15,
                          collapse=False)
  num_end_R = np.sum(samples[:, -1] == 3)
  assert num_end_R > (num_samples//2), (
    f"num_end_R should be most of num_samples ({num_samples}), but is only"
    f"{num_end_R}")

  q_param = np.array([0.99, 0.99, 0.99, 0.99])
  samples = util.sample_q(q_param, num_samples=num_samples, num_time=5,
                          collapse=False)
  num_end_R = np.sum(samples[:, -1] == 3)
  assert num_end_R < (num_samples//2), (
    f"num_end_R should be almost no samples, but found {num_end_R}")


def test_get_q_marginal_infected():
  num_time_steps = 3

  # Don't jump out of S state
  params = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  q_marginal = util.get_q_marginal_infected(np.array(params), num_time_steps)
  q_marginal_expected = np.zeros((num_time_steps))
  np.testing.assert_array_almost_equal(q_marginal, q_marginal_expected)

  # Jump immediately to I state and stay
  params = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  q_marginal = util.get_q_marginal_infected(np.array(params), num_time_steps)
  q_marginal_expected = np.array([0, 1, 1])
  np.testing.assert_array_almost_equal(q_marginal, q_marginal_expected)

  # Jump immediately to I state and jump out after
  params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  q_marginal = util.get_q_marginal_infected(np.array(params), num_time_steps)
  q_marginal_expected = np.array([0, 1, 0])
  np.testing.assert_array_almost_equal(q_marginal, q_marginal_expected)


def test_forward_propagate_q():
  num_time_steps = 23
  logits = np.random.randn(3*num_time_steps+1)
  params = util.sigmoid(logits)
  q_state = util.forward_propagate_q(params, num_time_steps)
  np.testing.assert_array_almost_equal(
    np.sum(q_state, axis=1), np.ones((num_time_steps)))


def test_d_penalty_term():
  contacts_all = [
    (0, 1, 2, [1]),
    (1, 0, 2, [1]),
    (3, 2, 2, [1]),
    (2, 3, 2, [1]),
    (4, 5, 2, [1]),
    (5, 4, 2, [1]),
    ]
  num_users = 6
  num_time_steps = 5

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  q_marginal_infected = np.array([
    [.8, .8, .8, .8, .8],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
  ])

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=.3,
    past_contacts=infect_counter.get_past_contacts_slice([user])[0],
    num_time_steps=num_time_steps)

  # Contact with user 0, which is infected with .8, so penalty for term should
  # be less (higher number) than no_term.
  assert np.all(d_term >= d_no_term)
  assert d_term[0] == 0
  assert d_no_term[0] == 0
  assert d_term[num_time_steps] == 0

  # Second test case
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1],
  ])

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=infect_counter.get_past_contacts_slice([user])[0],
    num_time_steps=num_time_steps)

  # With small p1, penalty for termination should be small (low number)
  assert np.all(d_term < 0.001)

  d_term_old, d_no_term_old = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=infect_counter.get_past_contacts_slice([user])[0],
    num_time_steps=num_time_steps)
  np.testing.assert_array_almost_equal(d_term, d_term_old)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_old)


def test_softmax():
  # Check that trick for numerical stability yields identical results

  def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

  logits = np.random.randn(13)

  np.testing.assert_array_almost_equal(
    softmax(logits),
    util.softmax(logits)
  )


def test_it_num_infected_probs():
  # For max entropy, all probs_total should be 2**-len(probs)
  probs = [.5, .5, .5, .5]
  for _, prob_total in util.it_num_infected_probs(probs):
    np.testing.assert_almost_equal(prob_total, 1/16)

  # For any (random) probs, the probs_total should sum to 1
  probs = list((random.random() for _ in range(5)))
  sums, probs_total = zip(*util.it_num_infected_probs(probs))
  assert len(list(sums)) == 2**5
  np.testing.assert_almost_equal(sum(probs_total), 1.0)

  # For one prob, should reduce to [(0, 1-p), (1, p)]
  prob = random.random()
  sums, probs_total = zip(*util.it_num_infected_probs([prob]))
  assert list(sums) == [0, 1]
  np.testing.assert_almost_equal(list(probs_total), [1-prob, prob])

  # Manual result
  probs = [.8, .7]
  expected = [(0, .2*.3), (1, .2*.7), (1, .8*.3), (2, .8*.7)]
  expected_sums, expected_probs_total = zip(*expected)
  result_sums, result_probs_total = zip(*util.it_num_infected_probs(probs))
  np.testing.assert_array_almost_equal(list(result_sums), list(expected_sums))
  np.testing.assert_array_almost_equal(
    list(result_probs_total), list(expected_probs_total))


def test_spread_buckets():
  num_sample_array = util.spread_buckets(100, 10)
  expected = 10 * np.ones((10))
  np.testing.assert_array_almost_equal(num_sample_array, expected)

  num_sample_array = util.spread_buckets(97, 97)
  np.testing.assert_array_almost_equal(num_sample_array, np.ones((97)))

  num_samples = np.sum(util.spread_buckets(100, 13))
  np.testing.assert_almost_equal(num_samples, 100)

  with np.testing.assert_raises(AssertionError):
    util.spread_buckets(13, 17)


def test_past_contact_array():
  contacts_all = [
    (1, 2, 4, [1]),
    (2, 1, 4, [1]),
    (0, 1, 4, [1])
    ]

  counter = util.InfectiousContactCount(
    contacts_all, None, num_users=6, num_time_steps=7)
  past_contacts = counter.get_past_contacts_slice([0, 1, 2])

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, 2+1, 3])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int16)


def test_enumerate_start_belief():
  seq_array = np.stack(list(
    util.iter_sequences(time_total=5, start_se=False)))

  start_belief = np.array([.1, .2, .3, .4])
  A_start_belief = util.enumerate_start_belief(seq_array, start_belief)

  expected = np.array(
    [0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

  np.testing.assert_array_almost_equal(A_start_belief, expected)


def test_update_beliefs():
  matrix = np.ones((3, 4))
  beliefs = np.zeros((2, 4))

  result = util.update_beliefs(
    matrix,
    beliefs,
    user_slice=np.array([0, 2], dtype=np.int32))
  expected = np.tile(np.array([0, 1, 0]), [4, 1]).T
  np.testing.assert_array_almost_equal(result, expected)

  result = util.update_beliefs(
    matrix,
    beliefs,
    user_slice=np.array([0, 2], dtype=np.int32),
    users_stale=[2])
  expected = np.tile(np.array([0, 1, 1]), [4, 1]).T
