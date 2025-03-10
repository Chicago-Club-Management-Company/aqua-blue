import numpy as np
import matplotlib.pyplot as plt
import aqua_blue

#################################################################################
#                              MACKEY-GLASS FUNCTION                            #   
#################################################################################

def mackey_glass(t0, tf, dt=0.01, beta=0.2, gamma=0.1, n=10, tau=17, x0=1.2): 
    steps = int((tf - t0) / dt)
    x = np.zeros(steps)
    x[0] = x0
    
    for i in range(1, steps):
        if i - tau / dt < 0:
            x_tau = 0.0
        else:
            x_tau = x[int(i - tau / dt)]
        
        x[i] = x[i-1] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[i-1])
    
    return x, np.arange(t0, tf, dt)

    
#################################################################################
#                              TIME SERIES GENERATION                           #   
#################################################################################

def main(): 

    x, t = mackey_glass(0, 1000, dt=0.001)
    
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=x, times=t)
    
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    
#################################################################################
#                               MODEL TRAINING                                  #
#################################################################################
    
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=1,
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)
    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)

#################################################################################
#                               MODEL EVALUATION                                #
#################################################################################

    actual_future = mackey_glass(prediction.times[0], prediction.times[-1],  dt=0.001)
    
    plt.plot(prediction.times, actual_future[0])
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual', 'predicted'], shadow=True)
    plt.title('Mackey-Glass System')
    plt.show()

if __name__ == "__main__":
    
    main()