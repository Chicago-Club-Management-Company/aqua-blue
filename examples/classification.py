from pathlib import Path
import tarfile
import random
from itertools import chain

import requests
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
import numpy as np

import aqua_blue


URL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
GZ_PATH = Path("speech-data.tar.gz")
DECOMPRESSED_PATH = Path("speech-data")

RESERVOIR_DIMENSIONALITY = 100
NUM_TRAINING_SAMPLES = 50
NUM_TESTING_SAMPLES = 10

CATEGORIES = {"left", "right"}
CLASSIFIERS = {category: i for i, category in enumerate(CATEGORIES)}


def download_dataset():

    response = requests.get(URL)
    with GZ_PATH.open("wb") as file:
        file.write(response.content)


def decompress_dataset():

    with tarfile.open(GZ_PATH) as file:
        file.extractall(DECOMPRESSED_PATH)


def wav_to_feature_vector(p: Path, model: aqua_blue.models.Model) -> np.typing.NDArray[np.floating]:

    sample_rate, data = wavfile.read(p)
    if not sample_rate == data.shape[0]:
        raise ValueError
    reservoir_states = np.zeros((data.shape[0], model.reservoir.reservoir_dimensionality))
    model.res_state = np.zeros(model.reservoir.reservoir_dimensionality)
    for i in range(1, data.shape[0]):
        reservoir_states[i, :] = model.reservoir.update_reservoir(data.reshape(-1, 1)[i - 1])

    return reservoir_states[1:].flatten()


def main():

    random.seed(0)

    model = aqua_blue.models.Model(
        reservoir=aqua_blue.reservoirs.DynamicalReservoir(
            reservoir_dimensionality=RESERVOIR_DIMENSIONALITY,
            input_dimensionality=1
        ),
        readout=aqua_blue.readouts.LinearReadout()
    )

    if not GZ_PATH.exists():
        download_dataset()

    if not DECOMPRESSED_PATH.exists():
        decompress_dataset()

    paths = list(chain.from_iterable(
        (DECOMPRESSED_PATH / category).glob("*.wav") for category in CLASSIFIERS
    ))

    training_features = []
    training_classifiers = []
    while len(training_features) < NUM_TRAINING_SAMPLES:
        p = random.choice(paths)
        try:
            feature_vector = wav_to_feature_vector(p, model)
        except ValueError:
            continue
        training_classifiers.append(CLASSIFIERS[p.parent.name])
        training_features.append(feature_vector)

    training_features = np.array(training_features)
    means = training_features.mean(axis=0)
    stds = training_features.std(axis=0)
    training_features = (training_features - means) / stds

    clf = LogisticRegression(random_state=0).fit(training_features, training_classifiers)

    testing_feature_vectors = []
    testing_classifiers = []
    while len(testing_feature_vectors) < NUM_TESTING_SAMPLES:
        p = random.choice(paths)
        try:
            feature_vector = wav_to_feature_vector(p, model)
        except ValueError:
            continue
        testing_classifiers.append(CLASSIFIERS[p.parent.name])
        testing_feature_vectors.append((feature_vector - means) / stds)

    prediction = clf.predict(testing_feature_vectors)
    actual = np.array(testing_classifiers)
    accuracy = (prediction == actual).mean()
    print(accuracy)


if __name__ == "__main__":

    main()
