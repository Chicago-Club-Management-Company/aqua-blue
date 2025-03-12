from io import BytesIO
from copy import deepcopy
from datetime import datetime, timedelta

import pytest
import numpy as np

from zoneinfo import ZoneInfo

from aqua_blue import time_series, utilities, reservoirs, readouts, models, datetimelikearray


@pytest.fixture
def cosine_sine_series():

    steps = list(range(10))
    
    dependent_variables = np.vstack((np.cos(steps), np.sin(steps))).T

    return time_series.TimeSeries(
        dependent_variable=dependent_variables,
        times=steps
    )

@pytest.fixture
def datetime_arr(): 
    times_ = datetimelikearray.DatetimeLikeArray.from_array(
        input_array=np.arange(
            np.datetime64('2021-01-01T00:00:00'), 
            np.datetime64('2021-01-20T00:00:00'), 
            np.timedelta64(1, 'D'), 
            dtype = 'datetime64[s]'
        ),
        tz=ZoneInfo("America/New_York")
    )

    return times_ 

@pytest.fixture
def datetime_series(): 
    times_ = datetimelikearray.DatetimeLikeArray.from_array(
        input_array=np.arange(
            np.datetime64('2021-01-01T00:00:00'), 
            np.datetime64('2021-01-20T00:00:00'), 
            np.timedelta64(1, 'D'), 
            dtype = 'datetime64[s]'
        ),
        tz=ZoneInfo("America/New_York")
    )
    
    steps = list(range(times_.size))

    dependent_variables = np.vstack((np.cos(steps), np.sin(steps))).T
    
    return time_series.TimeSeries(
        dependent_variable=dependent_variables, 
        times=times_
    )


def test_non_uniform_timestep_error():
    
    with pytest.raises(ValueError):
        _ = time_series.TimeSeries(dependent_variable=np.ones(10), times=np.logspace(0, 1, 10))


def test_zero_timestep_error():

    with pytest.raises(ValueError):
        _ = time_series.TimeSeries(dependent_variable=np.ones(10), times=np.zeros(10))


def test_can_save_and_load_time_series():
    
    t_original = time_series.TimeSeries(dependent_variable=np.ones(shape=(10, 2)), times=np.arange(10))
    with BytesIO() as buffer:
        t_original.save(buffer)
        buffer.seek(0)
        t_loaded = time_series.TimeSeries.from_csv(buffer)
    
    assert t_original == t_loaded


def test_normalizer_inversion(cosine_sine_series):

    normalizer = utilities.Normalizer()
    t_normalized = normalizer.normalize(cosine_sine_series)
    t_denormalized = normalizer.denormalize(t_normalized)

    assert cosine_sine_series == t_denormalized


def test_pinv_workaround(cosine_sine_series):

    model = models.Model(
        reservoir=reservoirs.DynamicalReservoir(input_dimensionality=2, reservoir_dimensionality=10),
        readout=readouts.LinearReadout()
    )
    model.train(cosine_sine_series)


def test_can_add_time_series(cosine_sine_series):

    sine_cosine_series = deepcopy(cosine_sine_series)
    sine_cosine_series.dependent_variable[:, [0, 1]] = sine_cosine_series.dependent_variable[:, [1, 0]]
    summed = cosine_sine_series + sine_cosine_series

    assert np.all(
        summed.dependent_variable == cosine_sine_series.dependent_variable + sine_cosine_series.dependent_variable
    )
    assert all(t1 == t2 for t1, t2 in zip(summed.times, cosine_sine_series.times))
    assert all(t1 == t2 for t1, t2 in zip(cosine_sine_series.times, sine_cosine_series.times))


def test_time_series_addition_num_timesteps_error(cosine_sine_series):

    second_time_series = deepcopy(cosine_sine_series)
    dt = cosine_sine_series.times[1] - cosine_sine_series.times[0]
    second_time_series.times = np.append(cosine_sine_series.times, cosine_sine_series.times[-1] + dt)

    with pytest.raises(ValueError):
        _ = cosine_sine_series + second_time_series


def test_time_series_addition_spanning_error(cosine_sine_series):

    second_time_series = deepcopy(cosine_sine_series)
    second_time_series.times = [t + 1.0 for t in cosine_sine_series.times]
    with pytest.raises(ValueError):
        _ = cosine_sine_series + second_time_series


def test_can_subtract_time_series(cosine_sine_series):

    second_time_series = deepcopy(cosine_sine_series)
    second_time_series.dependent_variable = 0.5 * cosine_sine_series.dependent_variable
    t = cosine_sine_series - second_time_series

    assert np.all(t.dependent_variable == (cosine_sine_series.dependent_variable - second_time_series.dependent_variable))
    assert np.all(t.times == cosine_sine_series.times) and np.all(cosine_sine_series.times == second_time_series.times)


def test_can_concatenate_time_series(cosine_sine_series):

    second_time_series = deepcopy(cosine_sine_series)
    dt = second_time_series.timestep
    num_steps = len(second_time_series)
    second_time_series.times = [cosine_sine_series.times[-1] + step * dt for step in range(1, num_steps + 1)]
    t = cosine_sine_series >> second_time_series

    assert np.all(t.dependent_variable == np.concatenate((cosine_sine_series.dependent_variable, second_time_series.dependent_variable)))
    assert np.all(t.times == np.concatenate((cosine_sine_series.times, second_time_series.times)))


def test_time_series_concatenation_overlap_error(cosine_sine_series):

    second_time_series = deepcopy(cosine_sine_series)

    with pytest.raises(ValueError):
        _ = cosine_sine_series >> second_time_series


def test_timeseries_slicing(cosine_sine_series):

    subset = cosine_sine_series[:2]
    assert np.all(subset.dependent_variable == cosine_sine_series.dependent_variable[:2])
    assert np.all(subset.times == cosine_sine_series.times[:2])
    with pytest.raises(IndexError):
        _ = cosine_sine_series[10]


def test_timeseries_slice_assignment():
    ts = time_series.TimeSeries(
        dependent_variable=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        times=np.array([0, 1, 2, 3])
    )
    new_ts = time_series.TimeSeries(
        dependent_variable=np.array([[9, 9], [8, 8]]),
        times=np.array([0, 1])
    )
    ts[:2] = new_ts
    assert np.all(ts.dependent_variable == np.array([[9, 9], [8, 8], [5, 6], [7, 8]]))


def test_datetime_time_series(cosine_sine_series):
    
    time_init = datetime(
        year=2025,
        month=3,
        day=1,
        hour=12,
        tzinfo=ZoneInfo("America/Chicago")
    )
    
    _ = time_series.TimeSeries(
        dependent_variable=cosine_sine_series.dependent_variable,
        times=[time_init + step * timedelta(days=1.0) for step in range(10)]
    )


def test_datetime_writetolist(datetime_arr):
    list_series = datetime_arr.to_list()
    
    time_init = datetime(2021, 1, 1, 0, 0, 0, tzinfo = ZoneInfo("America/New_York")) 
    interval =  timedelta(days=1.0)
    
    times = [time_init + step * interval for step in range(datetime_arr.size)]
    
    assert list_series == times 

def test_datetime_fileio(datetime_arr): 
    with BytesIO() as buffer:
        datetime_arr.to_file(buffer, tz=ZoneInfo("America/New_York"))
        buffer.seek(0)
        loaded_series = datetimelikearray.DatetimeLikeArray.from_fp(buffer, tz=ZoneInfo("America/New_York"), dtype='datetime64')
    
    assert loaded_series == datetime_arr

