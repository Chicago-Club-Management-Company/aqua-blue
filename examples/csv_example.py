import aqua_blue
from pathlib import Path

def main(): 
    
    goldstocks = aqua_blue.time_series.TimeSeries.from_csv(
        fp=Path('goldstocks.csv'), 
        time_col='DATE',
        times_dtype='datetime64[s]',
        dependent_var_cols=['X', 'Y', 'Z', 'A', 'B'],
    )

    normalizer = aqua_blue.utilities.Normalizer()
    normalized_time_series = normalizer.normalize(goldstocks)
    
    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=5
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )

    model.train(normalized_time_series)
    
    horizon = 100
    prediction = model.predict(horizon=horizon)
    prediction = normalizer.denormalize(prediction)
    
    concatenated = goldstocks >> prediction

    concatenated.save(
        fp=Path('predicted_goldstocks.csv'),
        header='DATE,X,Y,Z,A,B',
        fmt=('%s', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f')
    )

if __name__ == '__main__':
    main()
