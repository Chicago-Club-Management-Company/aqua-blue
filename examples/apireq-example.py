import requests
import aqua_blue
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import time

def main(): 
    req = requests.get("https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/01001099999.csv")

    time_col = "DATE"
    dependent_var_cols = ["TEMP"]
    
    with BytesIO() as file:
        txt = req.text
        file.write(txt.encode('utf-8'))
        file.seek(0)
    
        DATA = aqua_blue.time_series.TimeSeries.from_csv(
            fp=file,
            time_col=time_col,
            dependent_var_cols=dependent_var_cols,
            times_dtype='datetime64[s]',
            max_rows=87,
        )
    
    TRAIN_DATA = aqua_blue.time_series.TimeSeries(
        times=DATA.times[:82], 
        dependent_variable=DATA.dependent_variable[:82, :]
    )
    
    normalizer = aqua_blue.utilities.Normalizer()
    normalized_time_series = normalizer.normalize(TRAIN_DATA)
    
    seed = int(time.time())
    generator = np.random.default_rng(seed)
    print(seed)
    
    w_res = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 100)
    )
    w_in = generator.uniform(
        low=-0.5,
        high=0.5,
        size=(100, 1)
    )

    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            w_in = w_in,
            w_res = w_res,
            input_dimensionality=1,
        ),
        readout=aqua_blue.readouts.LinearReadout(1e-1)
    )
    
    model.train(normalized_time_series)
    
    horizon = 5
    prediction = model.predict(horizon=horizon)
    prediction = normalizer.denormalize(prediction)
    print(prediction.dependent_variable)
    
    concatenated = TRAIN_DATA >> prediction

    plt.plot(concatenated.times, concatenated.dependent_variable, label='Predicted Future')
    plt.plot(DATA.times, DATA.dependent_variable, label='Actual Future')
    plt.legend()
    plt.show()
    

if __name__ == '__main__': 
    main()
