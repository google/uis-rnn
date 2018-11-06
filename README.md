# UIS-RNN

## Overview

This is the library for the
*Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN)* algorithm,
corresponding to the paper
[Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719).

This open source implementation is slightly different than the internal one
which we used to produce the results in the paper, due to dependencies on
some internal libraries.

**This is not an official Google product.**

## Dependencies

This library depends on:

* python 3.5+
* numpy 1.15.1
* pytorch 0.4.0

## Tutorial

To get started, simply run this command:

```bash
python3 demo.py --train_iteration=20000
```

This will train a UIS-RNN model using `data/training_data.npz`,
then perform inference on `data/testing_data.npz`, and print the
inference results.

All algorithms are implemented as the `UISRNN` class. First, construct a
`UISRNN` object by:

```python
model = UISRNN(args)
```

Next, train the model by calling the `fit()` function:

```python
model.fit(args, train_sequence, train_cluster_id)
```

Once we are done with the training, we can run the trained model to perform
inference on new sequences by calling the `predict()` function:

```python
predicted_label = model.predict(args, test_sequence)
```

## Citations

Our paper is cited as:

```
@article{zhang2018fully,
  title={Fully Supervised Speaker Diarization},
  author={Zhang, Aonan and Wang, Quan and Zhu, Zhenyao and Paisley, John and Wang, Chong},
  journal={arXiv preprint arXiv:1810.04719},
  year={2018}
}
```
