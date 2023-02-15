"""Constants for the entire project."""

import numpy as np
from typing import List, Tuple, Union

colors = ['b', 'm', 'r', 'k']
state_names = ['S', 'E', 'I', 'R']

# Directory with ABM parameter files
ABM_HOME = "nttw/data/abm_parameters/"

# States of ABM simulator:
# * UNINFECTED = 0
# * PRESYMPTOMATIC = 1
# * PRESYMPTOMATIC_MILD = 2
# * ASYMPTOMATIC = 3
# * SYMPTOMATIC = 4
# * SYMPTOMATIC_MILD = 5
# * HOSPITALISED = 6
# * CRITICAL = 7
# * RECOVERED = 8
# * DEATH = 9
# * QUARANTINED = 10
# * QUARANTINE_RELEASE = 11
# * TEST_TAKE = 12
# * TEST_RESULT = 13
# * CASE = 14
# * TRACE_TOKEN_RELEASE = 15
# * N_EVENT_TYPES = 16
state_to_seir = np.array(
  [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4], dtype=np.int32)

# Global type definitions

# (user_u, user_v, timestep, [features])
Contact = Tuple[int, int, int, List[Union[float, int]]]

# (user_u, timestep, outcome)
Observation = Tuple[int, int, int]
