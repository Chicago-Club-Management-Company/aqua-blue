import aqua_blue

import numpy as np 

def main():
    t = np.arange(5_000) / 100
    y = np.vstack((np.cos(t) ** 2, np.sin(t))).T

    time_series = aqua_blue.time_series.TimeSeries(dependent_variable=y, times=t)

    generator = np.random.default_rng(seed=0)
    w_res = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 100)
    )
    w_in = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 2)
    )
    horizon = 1_000

    p_times = np.linspace(t[-1], t[-1] + horizon*np.diff(t)[0], horizon)

    mp = aqua_blue.hyper_opt.ModelParams(
        time_series=time_series,
        input_dimensionality=2, 
        reservoir_dimensionality=100,
        w_in=w_in,
        w_res=w_res, 
        horizon=horizon,
        actual_future=np.vstack((np.cos(p_times) ** 2, np.sin(p_times))).T
    )
    
    optimizer = aqua_blue.hyper_opt.Optimizer(fn=aqua_blue.hyper_opt.default_loss(mp), max_evals=100)
    best = optimizer.optimize()

    print(best)

if __name__ == '__main__':
    main()