import json

import aqua_blue


def main():

    # some string that is valid json
    json_str = """
    {
        "dependent_variable": [
            [2.0, -1.0],
            [1.5403023058681398, -0.1585290151921035],
            [0.5838531634528576, -0.09070257317431829],
            [0.010007503399554585, -0.8588799919401328],
            [0.34635637913638806, -1.7568024953079282],
            [1.2836621854632262, -1.9589242746631386],
            [1.9601702866503659, -1.2794154981989259],
            [1.7539022543433047, -0.34301340128121094],
            [0.8544999661913865, -0.010641753376618213],
            [0.08886973811532306, -0.5878815147582435]
        ],
        "times": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    """

    # turn into a TimeSeries instance
    time_series = aqua_blue.time_series.TimeSeries(**json.loads(json_str))

    # normalize and feed into model
    normalizer = aqua_blue.utilities.Normalizer()
    normalized_time_series = normalizer.normalize(time_series)

    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=100,
            input_dimensionality=2
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )
    model.train(normalized_time_series)

    # predict 5 more steps
    horizon = 5
    prediction = model.predict(horizon=horizon)
    prediction = normalizer.denormalize(prediction)

    # concatenate prediction and original time series, and print out a new json
    concatenated = time_series >> prediction

    # turn into a dictionary and json dump it
    print(json.dumps(concatenated.to_dict()))


if __name__ == "__main__":

    main()
