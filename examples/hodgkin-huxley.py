import numpy as np
import matplotlib.pyplot as plt
import aqua_blue

#################################################################################
#                             HODGKIN-HUXLEY FUNCTION                           #   
#################################################################################

def hodgkin_huxley(t: np.ndarray, I_ext: np.ndarray) -> np.ndarray:
    C_m = 1.0 
    g_Na = 120.0 
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    
    def alpha_n(V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(V):
        return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(V):
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(V):
        return 1 / (1 + np.exp(-(V + 35) / 10))
    
    V = -65.0
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))

    V_trace = np.zeros_like(t)  
    
    for i in range(len(t)):

        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)
        
        I_ion = I_ext[i] - I_Na - I_K - I_L
        
        V += I_ion / C_m * (t[1] - t[0])
        
        n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * (t[1] - t[0])
        m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * (t[1] - t[0])
        h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * (t[1] - t[0])
        
        V_trace[i] = V
    
    return V_trace
    
#################################################################################
#                            TIME SERIES GENERATION                             #
#################################################################################

def main():

    t = np.arange(0.0, 50.0, 0.01)  # time array, in ms
    I_ext = np.zeros_like(t)
    I_ext[1000:4000] = 10  # external current, in uA/cm^2

    V_trace = hodgkin_huxley(t, I_ext)
    
    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=V_trace, times=t)
    
    normalizer = aqua_blue.utilities.Normalizer()
    time_series = normalizer.normalize(time_series)
    
################################################################################
#                             MODEL TRAINING                                   #
################################################################################
    
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=1
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(time_series)

    prediction = model.predict(horizon=1_000)
    prediction = normalizer.denormalize(prediction)
    
################################################################################
#                             MODEL EVALUATION                                 #
################################################################################
    I_ext = np.zeros_like(prediction.times)
    I_ext[1000:4000] = 10 
    actual_future = hodgkin_huxley(prediction.times, I_ext)
    plt.plot(prediction.times, actual_future)
    plt.xlabel('t')
    plt.plot(prediction.times, prediction.dependent_variable)
    plt.legend(['actual', 'predicted'], shadow=True)
    plt.title('Hodgkin-Huxley System')
    plt.show()
    
if __name__ == "__main__":
    
    main()

 
