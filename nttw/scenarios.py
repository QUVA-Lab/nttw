"""Simulate data from different scenarios of observations and contacts.

General abstraction is that functions return a list of contacts and a list of
observations.
"""

import json
from nttw import data_gen
import numpy as np  # pylint: disable=unused-import
import os
import random


def make_plain_observations(obs):
  return [(o['u'], o['time'], o['outcome']) for o in obs]


def make_plain_contacts(contacts):
  return [
    (c['u'], c['v'], c['time'], [int(c['features'][0])]) for c in contacts]


def generate_scenario(scenario=2):  # pylint: disable=too-many-statements
  """Generates scenarios of data."""
  # pylint: disable=too-many-branches

  # Default parameters:
  num_time_steps = 35
  num_users = 5
  alpha = 0.001
  beta = 0.01
  p0 = 0.01
  p1 = .3

  # Construct Geometric distro's for E and I states
  g = 1/3
  h = 1/3

  states = None
  metadata = None

  if scenario == 0:
    # pylint: disable=unused-variable
    num_samples = 3000
    num_burn = 100

    p1 = 0.9
    # Times (in days) for simulated data
    t_test = 2
    t_touch01 = 1
    num_time_steps = 6
    t_touch02 = 4
    t_touch_after = 80
    observations_all = [
      # {'u': 0, 'time': t_test, 'outcome': 1},
      {'u': 1, 'time': t_test, 'outcome': 1}]
    contacts_all = [
      # {'u': 0, 'v': 1, 'time': t_touch01, 'features': [1]},
      {'u': 0, 'v': 1, 'time': t_touch01, 'features': [1]},
      {'u': 0, 'v': 1, 'time': 6, 'features': [1]},
      # {'u': 0, 'v': 2, 'time': t_touch02, 'features': [1]},
      # {'u': 2, 'v': 0, 'time': t_touch02, 'features': [1]},
      # {'u': 2, 'v': 3, 'time': t_touch_after, 'features': [1]},
      # {'u': 3, 'v': 2, 'time': t_touch_after, 'features': [1]},
      # {'u': 2, 'v': 4, 'time': t_touch_after, 'features': [1]},
      # {'u': 4, 'v': 2, 'time': t_touch_after, 'features': [1]}
      ]
    num_users = 2
    # pylint: enable=unused-variable
  elif scenario == 1:
    num_samples = 3000
    num_burn = 100

    p1 = 0.9
    # Times (in days) for simulated data
    num_time_steps = 5
    observations_all = [
      # {'u': 0, 'time': 3, 'outcome': 1},
      {'u': 1, 'time': 4, 'outcome': 1},
      ]
    contacts_all = [
      {'u': 0, 'v': 1, 'time': 1, 'features': [1]},
      # {'u': 1, 'v': 0, 'time': 1, 'features': [1]},
    ]
    num_users = 2
  elif scenario == 2:
    num_samples = 1000
    num_burn = 100

    # Get data from disk
    dir_data = "nttw/data"
    fname_obs = os.path.join(dir_data, "observations.json")
    fname_contacts = os.path.join(dir_data, "contacts.json")
    fname_metadata = os.path.join(dir_data, "metadata.json")
    fname_states = os.path.join(dir_data, "states.json")

    with open(fname_obs, 'r', encoding='UTF-8') as fp:
      observations_all = json.load(fp)

    with open(fname_contacts, 'r', encoding='UTF-8') as fp:
      contacts_all = json.load(fp)

    with open(fname_metadata, 'r', encoding='UTF-8') as fp:
      metadata = json.load(fp)

    with open(fname_states, 'rb') as fp:
      states = np.load(fp)["states"]

    print(f"Found {len(observations_all)} observations")

    # Load all from metadata
    num_time_steps = metadata["num_time_steps"]
    num_users = metadata["num_users"]
    alpha = metadata["alpha"]
    beta = metadata["beta"]
    p0 = metadata["p0"]
    p1 = metadata["p1"]
    g = metadata["g"]
    h = metadata["h"]

  elif scenario == 3:
    num_samples = 3000
    num_burn = 100

    p1 = 0.9
    # Times (in days) for simulated data
    num_time_steps = 5
    observations_all = [
      {'u': 0, 'time': 2, 'outcome': 1},
      {'u': 1, 'time': 3, 'outcome': 1},
      ]
    contacts_all = [
      {'u': 0, 'v': 2, 'time': 0, 'features': [1]},
      {'u': 1, 'v': 2, 'time': 1, 'features': [1]},
    ]
    num_users = 3

  elif scenario == 4:
    num_samples = 100
    num_burn = 20

    observations_all = [
      {'u': 0, 'time': 5, 'outcome': 0},
      {'u': 0, 'time': 10, 'outcome': 1},
      {'u': 0, 'time': 14, 'outcome': 1},
      {'u': 0, 'time': 15, 'outcome': 0},
      # --
      {'u': 1, 'time': 6, 'outcome': 0},
      {'u': 1, 'time': 10, 'outcome': 1},
      {'u': 1, 'time': 14, 'outcome': 1},
      {'u': 1, 'time': 16, 'outcome': 0},
      # --
      {'u': 2, 'time': 6, 'outcome': 0},
      {'u': 2, 'time': 7, 'outcome': 1},
      {'u': 2, 'time': 15, 'outcome': 1},
      {'u': 2, 'time': 16, 'outcome': 0},
      # --
      {'u': 3, 'time': 7, 'outcome': 0},
      {'u': 3, 'time': 8, 'outcome': 1},
      {'u': 3, 'time': 12, 'outcome': 1},
      {'u': 3, 'time': 13, 'outcome': 0},
      # --
      {'u': 4, 'time': 5, 'outcome': 0},
      {'u': 4, 'time': 7, 'outcome': 1},
      {'u': 4, 'time': 12, 'outcome': 1},
      {'u': 4, 'time': 13, 'outcome': 0},
      # --
      {'u': 5, 'time': 6, 'outcome': 0},
      {'u': 5, 'time': 7, 'outcome': 1},
      {'u': 5, 'time': 11, 'outcome': 1},
      {'u': 5, 'time': 14, 'outcome': 0},
    ]

    contacts_all = []
    num_users = 6
    num_time_steps = 20

  elif scenario == 5:
    num_samples = 100
    num_burn = 20

    interval = 5
    alpha, beta = 1E-5, 1E-5

    observations_all = [
      {'u': 0, 'time': t, 'outcome': 0} for t in range(interval)
    ] + [
      {'u': 0, 'time': t, 'outcome': 1} for t in range(interval, 2*interval)
    ] + [
      {'u': 0, 'time': t, 'outcome': 0} for t in range(2*interval, 3*interval)
    ]

    contacts_all = []
    num_users = 1
    num_time_steps = int(interval*4)

  elif scenario == 6:
    num_samples = 5000
    num_burn = 100
    p1 = 0.05
    t_test = 15
    num_users = 30

    it_contacts = data_gen.init_contacts(
      num_users=num_users,
      num_time_steps=num_time_steps,
      qIbar=1/h,
      R0=1.2,
      p1=p1,
    )
    contacts_all = []
    for contacts_per_day in it_contacts:
      for contact in contacts_per_day:
        contacts_all.append(
          {'u': contact[0],
           'v': contact[1],
           'time': contact[2],
           'features': [contact[3]]})

    observations_all = [
      {'u': u, 'time': t_test, 'outcome': 1} for u in range(3)]
    observations_all += [
      {'u': u+10, 'time': t_test+10, 'outcome': 1} for u in range(3)]

  elif scenario == 7:
    # pylint: disable=unused-variable
    num_samples = 3000
    num_burn = 100

    p1 = 0.9
    # Times (in days) for simulated data
    t_test = 3
    t_touch01 = 2
    num_time_steps = 8
    t_touch12 = 4
    t_touch_after = 80
    observations_all = [
      # {'u': 0, 'time': t_test, 'outcome': 1},
      {'u': 0, 'time': t_test, 'outcome': 1}]
    contacts_all = [
      # {'u': 0, 'v': 1, 'time': t_touch01, 'features': [1]},
      {'u': 0, 'v': 1, 'time': t_touch01, 'features': [1]},
      {'u': 1, 'v': 2, 'time': t_touch12, 'features': [1]},
      # {'u': 2, 'v': 0, 'time': t_touch02, 'features': [1]},
      # {'u': 2, 'v': 3, 'time': t_touch_after, 'features': [1]},
      # {'u': 3, 'v': 2, 'time': t_touch_after, 'features': [1]},
      # {'u': 2, 'v': 4, 'time': t_touch_after, 'features': [1]},
      # {'u': 4, 'v': 2, 'time': t_touch_after, 'features': [1]}
      ]
    num_users = 3
    # pylint: enable=unused-variable

  elif scenario == 8:
    # pylint: disable=unused-variable
    num_samples = 3000
    num_burn = 100

    alpha = 0.00001

    p0 = 0.01
    p1 = 0.99
    num_time_steps = 20
    observations_all = [
      {'u': 0, 'time': 6, 'outcome': 1}]
    contacts_all = [
      {'u': 0, 'v': 1, 'time': 5, 'features': [1]},
      {'u': 1, 'v': 2, 'time': 8, 'features': [1]},
      {'u': 2, 'v': 3, 'time': 11, 'features': [1]},
      {'u': 3, 'v': 4, 'time': 14, 'features': [1]},
      ]
    num_users = 5
    # pylint: enable=unused-variable

  else:
    num_samples = 500
    num_burn = 100
    num_users = 5

    t_test_neg = 5
    t_test_pos = 12
    t_test_neg_r = 20

    observations_all = []

    for user in range(num_users):
      observations_all += [
        {'u': user, 'time': t_test_neg+random.randrange(0, 3), 'outcome': 0},
        {'u': user, 'time': t_test_pos+random.randrange(0, 3), 'outcome': 1},
        {'u': user, 'time': t_test_neg_r+random.randrange(0, 3), 'outcome': 0}]

    contacts_all = []

  if states is None:
    states = -1 * np.ones((num_users, num_time_steps))

  return (contacts_all, observations_all, states, g, h, num_samples, num_burn,
          num_users, num_time_steps, alpha, beta, p0, p1, metadata)
