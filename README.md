# UIS-RNN

## Overview

This is the library for the
*Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN)* algorithm.
UIS-RNN solves the problem of segmenting and clustering sequential data
by learning from examples.

This algorithm was originally proposed in the paper
[Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719).

The work has been introduced by
[Google AI Blog](https://ai.googleblog.com/2018/11/accurate-online-speaker-diarization.html).

![gif](resources/uisrnn.gif)

## Disclaimer

This open source implementation is slightly different than the internal one
which we used to produce the results in the
[paper](https://arxiv.org/abs/1810.04719), due to dependencies on
some internal libraries.

We CANNOT share the data, code, or model for the speaker recognition system
(d-vector embeddings) used in the paper, since the speaker recognition system
heavily depends on Google's internal infrastructure and proprietary data.

**This library is NOT an official Google product.**

## Dependencies

This library depends on:

* python 3.5+
* numpy 1.15.1
* pytorch 0.4.0
* scipy 1.1.1 (for evaluation only)

## Getting Started

### Run the demo

To get started, simply run this command:

```bash
python3 demo.py --train_iteration=20000
```

This will train a UIS-RNN model using `data/training_data.npz`,
then store the model on disk, perform inference on `data/testing_data.npz`,
print the inference results, and save the approximate accuracy in a text file.

PS. The files under `data/` are manually generated *toy data*,
for demonstration purpose only.
These data are very simple, so we are supposed to get 100% accuracy on the
testing data.

### Run the tests

You can also verify the correctness of this library by running:

```bash
sh run_tests.sh
```

If you fork this library and make local changes, be sure to use these tests
as a sanity check.

Besides, these tests are also great examples for learning
the APIs, especially `tests/integration_test.py`.

## Core APIs

### Glossary

| General Machine Learning | Speaker Diarization    |
|--------------------------|------------------------|
| Sequence                 | Utterance              |
| Observation              | Embedding / d-vector   |
| Label / Cluster ID       | Speaker                |

### Model construction

All algorithms are implemented as the `UISRNN` class. First, construct a
`UISRNN` object by:

```python
model = UISRNN(args)
```

The definitions of the args are described in `model/arguments.py`.

### Training

Next, train the model by calling the `fit()` function:

```python
model.fit(train_sequence, train_cluster_id, args)
```

Here `train_sequence` should be a 2-dim numpy array of type `float`, for
the **concatenated** observation sequences. For speaker diarization, this
could be the [d-vector embeddings](https://arxiv.org/abs/1710.10467).

For example, if you have *M* training utterances,
and each utterance is a sequence of *L* embeddings. Each embedding is
a vector of *D* numbers. The the shape of `train_sequence` is *N * D*,
where *N = M * L*.

`train_cluster_id` is a 1-dim list or numpy array of strings, of length *N*.
It is the **concatenated** ground truth labels of all training data. For
speaker diarization, these labels are the speaker identifiers for each
observation (*e.g.* d-vector).

Since we are concatenating observation sequences, it is important to note that,
ground truth labels in `train_cluster_id` across different sequences are
supposed to be **globally unique**.

For example, if the set of labels in the first
sequence is `{'A', 'B', 'C'}`, and the set of labels in the second sequence
is `{'B', 'C', 'D'}`. Then after concatenation, we should rename them to
something like `{'1_A', '1_B', '1_C'}` and `{'2_B', '2_C', '2_D'}`,
unless `'B'` and `'C'` in the two sequences are meaningfully identical
(in speaker diarization, this means they are the same speakers across
utterances).

The definitions of the args are described in `model/arguments.py`.

### Prediction

Once we are done with the training, we can run the trained model to perform
inference on new sequences by calling the `predict()` function:

```python
predicted_label = model.predict(test_sequence, args)
```

Here `test_sequence` should be a 2-dim numpy array of type `float`,
corresponding to a **single** observation sequence.

The returned `predicted_label` is a list of integers, with the same
length as `test_sequence`.

The definitions of the args are described in `model/arguments.py`.

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
