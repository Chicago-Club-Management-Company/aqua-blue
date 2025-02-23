__version__ = "0.0.20"
__authors__ = [
    "Jacob Jeffries",
    "Ameen Mahmood",
    "Avik Thumati",
    "Samuel Josephs"
]
__author_emails__ = [
    "jacob.jeffries@chicagoclubteam.org",
    "ameen.mahmood@chicagoclubteam.org",
    "avik.thumati@chicagoclubteam.org",
    "samuel.joseph@chicagoclubteam.org"
]
__url__ = "https://github.com/Chicago-Club-Management-Company/aqua-blue"

from .reservoir import EchoStateNetwork as EchoStateNetwork
from .reservoir import InstabilityWarning as InstabilityWarning
from .time_series import TimeSeries as TimeSeries
from .time_series import ShapeChangedWarning as ShapeChangedWarning

from . import utilities as utilities
from . import reservoirs as reservoirs
from . import readouts as readouts
from . import models as models
from . import time_series as time_series
