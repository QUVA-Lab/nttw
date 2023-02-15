"""Unit tests for simulator.py"""

from nttw import simulator


def test_window_cut():

  contacts = [
    (0, 23, 0, [1]),
    (0, 23, 1, [1]),
    (0, 23, 2, [1]),
    (0, 23, 3, [1]),
    (0, 23, 4, [1]),
    (0, 23, 5, [1]),
    (0, 23, 6, [1]),
    (0, 23, 7, [1]),
    (0, 23, 8, [1]),
    (0, 23, 9, [1]),
  ]

  sim = simulator.CRISPSimulator(num_time_steps=30, num_users=100, params={})
  sim.init_day0(contacts=contacts)

  assert sim.get_contacts() == contacts
  assert len(sim.get_contacts()) == 10

  sim.set_window(days_offset=1)

  assert sim.get_contacts() == contacts[:-1]
  assert len(sim.get_contacts()) == 9

  sim.set_window(days_offset=5)

  assert sim.get_contacts() == contacts[:-5]
  assert len(sim.get_contacts()) == 5

  sim.set_window(days_offset=6)

  assert sim.get_contacts() == contacts[:-6]
  assert len(sim.get_contacts()) == 4


def test_abm_simulator():

  num_time_steps = 100
  num_users = 10000

  params = {
    "p0": 0.001,
    "p1": .1,
    "g": .2,
    "h": .2,
  }

  sim = simulator.ABMSimulator(num_time_steps, num_users, params)
  sim.init_day0(contacts=[])

  sim.step()
  sim.step()
  assert sim.get_current_day() == 2
