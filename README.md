# Audio Effect Replicator

Replicate audio effects by machine learning.

## usage

```sh
# train model
python train.py -c config.yml

# predict with trained model
python predict.py -c config.yml -i input.wav -o predicted.wav -m checkpoint/20180208_235128/model_000031.h5
```
