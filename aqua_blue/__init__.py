r"""
.. include:: ../README.md

# Examples

## 🐇 Basic Lotka-Volterra example

Below is an example of using `aqua-blue` to predict the predator-prey
[Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):

$$ \dot x = \alpha x - \beta xy $$
$$ \dot y = -\gamma y + \delta xy$$

with parameters $\beta = 0.02$, $\gamma = 0.3$, and $\delta = 0.01$, and initial conditions
$(x_0, y_0) = (20, 9)$. We train a reservoir computer with a reservoir dimensionality of $1000$ over $0\leq t\leq 10$,
with $1000$ timesteps. Then, we predict the next $1000$ timesteps.

Here, we use `scipy.integrate.solve_ivp` to integrate the system of differential equations.


```py
.. include:: ../examples/lotka-volterra.py
```

## 🕓 Using datetime objects

Below is an example of a simple sine-cosine task similar to above, using `datetime.datetime` objects as times.

```py
.. include:: ../examples/sine-cosine.py
```

## 📡 Load and output a JSON string

Below is an example of inputting a `json` string as the training data, and outputting a `json` string for the
prediction. This is particularly useful for interfacing `aqua-blue` with already-existing systems.

```py
.. include:: ../examples/json_example.py
```
"""

__version__ = "0.2.8"
__authors__ = [
    "Jacob Jeffries",
    "Hrishikesh Belagali",
    "Avik Thumati",
    "Ameen Mahmood",
    "Samuel Josephs",
]
__author_emails__ = [
    "jacob.jeffries@ccmc.global",
    "hrishikesh.belagali@ccmc.global",
    "avik.thumati@ccmc.global",
    "ameen.mahmood@ccmc.global",
    "samuel.josephs@ccmc.global"
]
__url__ = "https://github.com/Chicago-Club-Management-Company/aqua-blue"

from . import utilities as utilities
from . import reservoirs as reservoirs
from . import readouts as readouts
from . import models as models
from . import time_series as time_series
