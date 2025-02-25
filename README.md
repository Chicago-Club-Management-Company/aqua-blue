# aqua-blue
Lightweight and basic reservoir computing library

[![Custom shields.io](https://img.shields.io/badge/docs-brightgreen?logo=github&logoColor=green&label=gh-pages)](https://chicago-club-management-company.github.io/aqua-blue/)


[![PyPI version shields.io](https://img.shields.io/pypi/v/aqua-blue.svg)](https://pypi.python.org/pypi/aqua-blue/)
[![PyPI pyversions shields.io](https://img.shields.io/pypi/pyversions/aqua-blue.svg)](https://pypi.python.org/pypi/aqua-blue/)

## 🌊 What is aqua-blue?

`aqua-blue` is a lightweight `python` library for reservoir computing (specifically [echo state networks](https://en.wikipedia.org/wiki/Echo_state_network)) depending only on `numpy`. `aqua-blue`'s namesake comes from:

- A blue ocean of data, aka a reservoir 💧
- A very fancy cat named Blue 🐾

## 📥 Installation

`aqua-blue` is on PyPI, and can therefore be installed with `pip`:

```bash
pip install aqua-blue
```

## 📝 Quickstart

```py
import numpy as np
import aqua_blue

# generate arbitrary two-dimensional time series
# y_1(t) = cos(t), y_2(t) = sin(t)
# resulting dependent variable has shape (number of timesteps, 2)
t = np.linspace(0, 4.0 * np.pi, 10_000)
y = np.vstack((2.0 * np.cos(t) + 1, 5.0 * np.sin(t) - 1)).T

# create time series object to feed into echo state network
time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)

# normalize
normalizer = aqua_blue.utilities.Normalizer()
time_series = normalizer.normalize(time_series)

# make model and train
model = aqua_blue.models.Model(
    reservoir=aqua_blue.reservoirs.DynamicalReservoir(
        reservoir_dimensionality=100,
        input_dimensionality=2
    ),
    readout=aqua_blue.readouts.LinearReadout()
)
model.train(time_series)

# predict and denormalize
prediction = model.predict(horizon=1_000)
prediction = normalizer.denormalize(prediction)
```

## 📃 License

`aqua-blue` is released under the MIT License.

---

![Blue](https://raw.githubusercontent.com/Chicago-Club-Management-Company/aqua-blue/refs/heads/main/assets/blue.jpg)

*Blue, the cat behind `aqua-blue`.*