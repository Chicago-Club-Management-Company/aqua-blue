import numpy as np
import matplotlib.pyplot as plt

import datetime
from zoneinfo import ZoneInfo

import aqua_blue

def main():
    
    start_date = datetime.datetime.now().astimezone(ZoneInfo("Indian/Maldives"))

    # Generate 10,000 datetime objects, each 1 minute apart
    t = [start_date + datetime.timedelta(minutes=i) for i in range(10000)]
    a = np.arange(len(t))/100
    y = np.vstack((np.cos(a)+1, np.sin(a)-1)).T
    time_series = aqua_blue.TimeSeries(dependent_variable=y, times=t)
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)

    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)
    

    actual_future = np.vstack((
        (np.cos(np.arange(a[-1], a[-1]+10, step=0.01))+1, np.sin(np.arange(a[-1], a[-1]+10, step=0.01))-1)
    )).T
    
    root_mean_square_error = np.sqrt(
        np.mean((actual_future - prediction.dependent_variable) ** 2)
    )
    
    print(root_mean_square_error)
    plt.plot(prediction.times, actual_future)
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual_x', 'actual_y', 'predicted_x', 'predicted_y'])
    plt.show()

if __name__ == "__main__":

    main()