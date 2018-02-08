from argparse import ArgumentParser
import yaml
import numpy as np
from keras.models import load_model
from fx_replicator import (
    build_model, load_wave, save_wave, sliding_window, LossFunc
)

def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)

    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]

    data = load_wave(args.input_file)

    # padding and rounded up to the batch multiple
    block_size = output_timesteps * batch_size
    prepad = input_timesteps - output_timesteps
    postpad = len(data) % block_size
    padded = np.concatenate((
        np.zeros(prepad, np.float32),
        data,
        np.zeros(postpad, np.float32)))
    x = sliding_window(padded, input_timesteps, output_timesteps)
    x = x[:, :, np.newaxis]

    model = load_model(
        args.model_file,
        custom_objects={"LossFunc": LossFunc(output_timesteps)})
    
    y = model.predict(x, batch_size=batch_size)
    y = y[:, -output_timesteps:, :].reshape(-1)[:len(data)]
    save_wave(y, args.output_file)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    parser.add_argument(
        "--input_file", "-i",
        help="input wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--output_file", "-o", default="./predicted.wav",
        help="output wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--model_file", "-m",
        help="input model file (*.h5)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
