r"""
.. include:: ../README.md

# Examples

## 🐇 Basic Lotka-Volterra example

Below is an example of using `aqua-blue` to predict the predator-prey
[Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):

$$ \dot x = \alpha x - \beta xy $$
$$ \dot y = -\gamma y + \delta xy$$

with parameters $\alpha = 0.1$, $\beta = 0.02$, $\gamma = 0.3$, and $\delta = 0.01$, and initial conditions
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
.. include:: ../examples/json-example.py
```

## 🏋 Explicit weights

Below is an example of generating explicit matrices for $W_\text{in}$ and $W_\text{res}$. Here, `sparsity=0.99` and
`spectral_radius=1.2` respectively zero-out $99\%$ of $W_\text{res}$'s elements and force $W_\text{res}$ to have a
[spectral radius](https://en.wikipedia.org/wiki/Spectral_radius) of $1.2$. We also showcase the `>>` operator, which
concatenates instances of `aqua_blue.time_series.TimeSeries`.

```py
.. include:: ../examples/explicit-weights.py
```

## 📈 Explicit activation function

Below is an example of using a different activation function to map from the input state to the reservoir.
Here, we use both hyperbolic tangent (`tanh`) and the [error function](https://en.wikipedia.org/wiki/Error_function)
(`erf`), and compare the results.

```py
.. include:: ../examples/activation-functions.py
```

## 🔗 HTTP Requests

Below is an example of pulling csv file data from a resource URL using the [requests](https://pypi.org/project/requests/) library. 
Here, we retrieve a time series of temperature data from [NCEI NOAA](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/01001099999.csv)
and use it for training and predicting temperatures.

```py
.. include:: ../examples/apireq-example.py
```

## 📁 Read from and write to CSV files

Below is an example of parsing data from a csv file (`goldstocks.csv`) and writing it to a `TimeSeries` object, which is used for training and predictions. 
The predictions are written to a new csv file (`predicted-goldstocks.csv`).

```py
.. include:: ../examples/csv-example.py
```
"""

__version__ = "0.2.14"
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
