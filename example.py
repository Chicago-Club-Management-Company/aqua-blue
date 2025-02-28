
import numpy as np
import aqua_blue
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from loktavolterra import lv
def main():
    
    y = lv(0, 10, 1000)
    t = np.linspace(0, 10, 1000)
    
    # create time series object to feed into echo state network
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)
    
    # normalize
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    
    # make model
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
    
    actual_future = lv(prediction.times[0], prediction.times[-1], 1_000)
    
    plt.plot(prediction.times, actual_future)
    plt.xlabel('t')
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual_x', 'actual_y', 'predicted_x', 'predicted_y'], shadow=True)
    plt.title('Lotka-Volterra System')
    plt.show()
if __name__ == "__main__":

    main()
