# Taken from https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python/blob/main/Chapter_07.ipynb
import numpy as np
import pandas as pd
from scipy import stats

from dowhy import CausalModel

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

COLORS = ["#00B0F0", "#FF0000"]


# First, we'll build a structural causal model (SCM)
class GPSMemorySCM:

    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.u_x = stats.truncnorm(0, np.infty, scale=5)
        self.u_y = stats.norm(scale=2)
        self.u_z = stats.norm(scale=2)
        self.u = stats.truncnorm(0, np.infty, scale=4)

    def sample(self, sample_size=100, treatment_value=None):
        """Samples from the SCM"""
        if self.random_seed:
            np.random.seed(self.random_seed)

        u_x = self.u_x.rvs(sample_size)
        u_y = self.u_y.rvs(sample_size)
        u_z = self.u_z.rvs(sample_size)
        u = self.u.rvs(sample_size)

        if treatment_value:
            gps = np.array([treatment_value] * sample_size)
        else:
            gps = u_x + 0.7 * u

        hippocampus = -0.6 * gps + 0.25 * u_z
        memory = 0.7 * hippocampus + 0.25 * u

        return gps, hippocampus, memory

    def intervene(self, treatment_value, sample_size=100):
        """Intervenes on the SCM"""
        return self.sample(treatment_value=treatment_value, sample_size=sample_size)


# Instantiate the SCM
scm = GPSMemorySCM()

# Generate observational data
gps_obs, hippocampus_obs, memory_obs = scm.sample(1000)

# Encode as a pandas df
df = pd.DataFrame(
    np.vstack([gps_obs, hippocampus_obs, memory_obs]).T, columns=["X", "Z", "Y"]
)


# Create the graph describing the causal structure
gml_graph = """
graph [
    directed 1
    
    node [
        id "X" 
        label "X"
    ]    
    node [
        id "Z"
        label "Z"
    ]
    node [
        id "Y"
        label "Y"
    ]
    node [
        id "U"
        label "U"
    ]
    
    edge [
        source "X"
        target "Z"
    ]
    edge [
        source "Z"
        target "Y"
    ]
    edge [
        source "U"
        target "X"
    ]
    edge [
        source "U"
        target "Y"
    ]
]
"""

# With graph
model = CausalModel(data=df, treatment="X", outcome="Y", graph=gml_graph)

# Identify the causal effect
estimand = model.identify_effect()
print(estimand)

# Estimate the causal effect
estimate = model.estimate_effect(
    identified_estimand=estimand, method_name="frontdoor.two_stage_regression"
)

print(f"Estimate of causal effect (linear regression): {estimate.value}")

# Run refutation tests
refute_subset = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.4,
)
