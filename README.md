# UIS-RNN

## Overview

This is the library for the Unbounded Interleaved-State Recurrent Neural Network
(UIS-RNN) algorithm, corresponding to the paper
[Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719).

This is not an official Google product.

## Dependencies

This library depends on:

* python 3.5+
* numpy 1.15.1
* pytorch 0.4.0

## Tutorial

To get started, simpy run this command:

```bash
python3 demo.py
```

This will train a UIS-RNN model using `data/training_data.npz`,
then perform inference on `data/testing_data.npz`, and print the
inference results.
