"""Simulating individuals in a pandemic with SEIR states."""
from abc import ABC
from COVID19 import model as abm_model
from COVID19 import simulation
import covid19

from nttw import constants, logger, util
from nttw.experiments import prequential
import numpy as np
import os
from typing import Any, Dict, List, Union


def plain_to_embedded_contacts(contact_list: List[List[int]]):
  """Converts plain contacts to embedded contacts."""

  def _embed(contact_tuple):
    # Tuples from ABM simulator have semantics:
    # (user_from, user_to, timestep, features)
    # TODO replace with 'contact_tuple[3]'
    return (contact_tuple[0], contact_tuple[1], contact_tuple[2], [1.])
  yield from map(_embed, contact_list)


class Simulator(ABC):
  """Base class for a simulator."""

  def __init__(
      self,
      num_time_steps: int,
      num_users: int,
      params: Dict[str, Any]) -> None:
    self.num_time_steps = num_time_steps
    self.num_users = num_users
    self.params = params

    self._day_current = 0
    self.states = None  # Type will depend on implementation

    # Note that contacts are offset with self._day_start_window and contacts
    # prior to self._day_start_window have been discarded.
    self._day_start_window = 0
    # List of (user, timestep, outcome), all integers
    self._observations_all = []
    # List of (user_u, user_v, timestep, [features])
    self._contacts = []

  def set_window(self, days_offset: int):
    """Sets the window with days_offset at day0.

    All days will start counting 0 at days_offset.
    The internal counter self._day_start_window keeps track of the previous
    counting for day0.
    """
    to_cut_off = max((0, days_offset - self._day_start_window))
    self._observations_all = list(prequential.offset_edges(
      self._observations_all, to_cut_off))
    self._contacts = list(prequential.offset_edges(
      self._contacts, to_cut_off))
    self._day_start_window = days_offset

  def get_current_day(self) -> int:
    """Returns the current day in absolute counting.

    Note, this day number is INDEPENDENT of the windowing.
    """
    # 0-based indexes!
    return self._day_current

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return np.zeros((self.num_users))

  def get_contacts(self) -> List[constants.Contact]:
    """Returns contacts.

    Note that contacts are offset with self._day_start_window and contacts prior
    to self._day_start_window have been discarded.
    """
    return self._contacts

  def get_observations_today(
      self,
      users_to_observe: List[int],
      p_obs_infected: Union[List[float], np.ndarray],
      p_obs_not_infected: Union[List[float], np.ndarray]
      ) -> List[constants.Observation]:
    """Returns the observations for current day."""

    day_relative = self.get_current_day() - self._day_start_window
    observations_new = list(prequential.get_observations_one_day(
      self.get_states_today(), users_to_observe, day_relative,
      p_obs_infected, p_obs_not_infected))
    self._observations_all += observations_new
    return observations_new

  def get_observations_all(self) -> List[Dict[str, int]]:
    return self._observations_all

  def step(self, num_steps: int = 1):
    self._day_current += num_steps

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """


class CRISPSimulator(Simulator):
  """Simulator based on generative code with CRISP paper."""

  def init_day0(self, contacts: List[constants.Contact]):
    self._contacts = contacts
    states, obs_list_current = prequential.init_states_observations(
      self.num_users, self.num_time_steps)

    self.states = states  # np.ndarray in size [num_users, num_timesteps]
    self._observations_all = obs_list_current  # List of dictionaries

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return self.states[:, self._day_current]

  def step(self, num_steps: int = 1):
    self._day_current += num_steps

    # Set contact days as simulate_one_day works with absolute times.
    contacts = list(
      prequential.delay_contacts(self._contacts, self._day_start_window))
    self.states = prequential.simulate_one_day(
      self.states, contacts, self._day_current,
      self.params["p0"], self.params["p1"], self.params["g"], self.params["h"])

    assert np.sum(self.states[:, self._day_current+1:]) == 0

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """
    # Make quarantines relative to window
    start_quarantine = self.get_current_day() + 1 - self._day_start_window

    self._contacts = prequential.remove_quarantine_users(
      self._contacts, users_to_quarantine, start_quarantine,
      t_delta=num_days)


class ABMSimulator(Simulator):
  """Simulator based on Oxford ABM."""

  def __init__(
      self,
      num_time_steps: int,
      num_users: int,
      params: Dict[str, Any]
      ) -> None:
    super().__init__(num_time_steps, num_users, params)

    filename = "baseline_parameters.csv"
    filename_hh = "baseline_household_demographics.csv"

    input_param_file = os.path.join(constants.ABM_HOME, filename)
    input_households = os.path.join(constants.ABM_HOME, filename_hh)

    util.check_exists(input_param_file)
    util.check_exists(input_households)

    logger.info("Construct ABM simulator")
    util.check_exists("results/tmp")
    params = abm_model.Parameters(
      input_param_file=input_param_file,
      param_line_number=1,
      output_file_dir="results/tmp/",
      input_households=input_households
    )

    if num_users < 10000:
      # TODO figure out why this fails
      logger.debug('ABM simulator might fail with <10k users')

    # Start with sufficient amount of initial infections. Start in E-state
    n_seed = 50 if num_users > 250000 else 5
    params.set_param("n_total", num_users)
    params.set_param("n_seed_infection", n_seed)

    model_init = abm_model.Model(params)
    self.model = simulation.COVID19IBM(model=model_init)
    self.sim = simulation.Simulation(env=self.model, end_time=num_time_steps)
    logger.info("Finished constructing ABM simulator")

  def init_day0(self, contacts: Any):
    del contacts
    # First call to steps will init day 0
    self.sim.steps(1)

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return np.take(
      constants.state_to_seir,
      np.array(covid19.get_state(self.model.model.c_model)))

  def step(self, num_steps: int = 1):
    self.sim.steps(num_steps)
    self._day_current += 1

    contacts_incoming = list(plain_to_embedded_contacts(
      covid19.get_contacts_daily(
        self.model.model.c_model, self._day_current)))

    self._contacts += prequential.offset_edges(
      contacts_incoming, self._day_start_window)

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """
    # Timestep of the actual ABM simulator could be found at
    #   * self.model.model.c_model.time
    covid19.intervention_quarantine_list(
      self.model.model.c_model,
      list(users_to_quarantine),
      self.get_current_day()+1 + num_days)
