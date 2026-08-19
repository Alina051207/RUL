# N-CMAPSS DS02-006 RUL Baselines

Baseline implementations for Remaining Useful Life (RUL) prediction on the **N-CMAPSS DS02-006** aircraft engine run-to-failure dataset.

This repository is intended to provide clean, reproducible **data-driven baseline models** before introducing more advanced fusion, physics-informed, or hybrid mechanisms.

## Overview

The current baseline set contains three representative sequence-learning approaches:

|Model|Main idea|Temporal modeling|
|-|-|-|
|**CNN**|Extract local degradation patterns from the input window|Convolutional feature extraction|
|**LSTM**|Model sequential degradation dynamics|Recurrent sequence modeling|
|**Transformer**|Model long-range temporal dependencies with self-attention|Transformer encoder|

All baselines use the same core prediction task:

> \*\*Input:\*\* flight conditions `W` + observable sensors `Xs`  
> \*\*Output:\*\* RUL `Y`

The resulting input dimension is **18** (4 flight-condition variables + 14 observable sensor variables).

\---

## Dataset

The project uses the NASA **N-CMAPSS** dataset, which simulates aircraft engine degradation under realistic flight conditions. The dataset combines flight-condition variables, observable sensor measurements, virtual sensor variables, engine identifiers, and RUL labels.

For the baseline models, only the directly usable flight-condition and observable-sensor information is used:

* `W`: 4 flight-condition variables
* `Xs`: 14 observable sensor variables
* `Y`: Remaining Useful Life (RUL)
* `A\[:, 0]`: engine/unit identifier used to construct independent temporal windows

The virtual sensors `Xv` and hidden health parameters are **not used as model inputs** in these baselines.

### Dataset structure

The original `.h5` file contains groups such as:

```text
A\_dev, A\_test, A\_var
T\_dev, T\_test, T\_var
W\_dev, W\_test, W\_var
X\_s\_dev, X\_s\_test, X\_s\_var
X\_v\_dev, X\_v\_test, X\_v\_var
Y\_dev, Y\_test
```

`dev` is used for training/validation, while `test` contains held-out engines.

### Input variables

|Variable|Description|
|-|-|
|`alt`|Altitude|
|`Mach`|Flight Mach number|
|`TRA`|Throttle-resolver angle|
|`T2`|Total temperature at fan inlet|
|`Wf`|Fuel flow|
|`Nf`|Physical fan speed|
|`Nc`|Physical core speed|
|`T24`|Total temperature at LPC outlet|
|`T30`|Total temperature at HPC outlet|
|`T48`|Total temperature at HPT outlet|
|`T50`|Total temperature at LPT outlet|
|`P15`|Total pressure in bypass duct|
|`P2`|Total pressure at fan inlet|
|`P21`|Total pressure at fan outlet|
|`P24`|Total pressure at LPC outlet|
|`Ps30`|Static pressure at HPC outlet|
|`P40`|Total pressure at burner outlet|
|`P50`|Total pressure at LPT outlet|

\---

## Experimental protocol

To avoid information leakage between engines, the development set is divided by engine unit rather than randomly by individual samples.

For the current DS02-006 experiments:

```text
Training engines : 2, 5, 10, 16, 18
Validation engine: 20
Test engines     : 11, 14, 15
```

A sliding window of **50 time steps** is used to construct each training example. The target is the RUL value at the final time step of the window.

Feature normalization is fitted on the training engines and then applied to validation and test data.

\---

## Baseline models

### 1\. CNN Baseline

The CNN baseline treats each temporal window as a structured time-feature representation and uses convolutional blocks to extract local degradation patterns.

Main settings:

* Input dimension: `18`
* Window size: `50`
* Four convolutional blocks
* Channels: `16 → 32 → 64 → 128`
* Batch normalization + ReLU
* Temporal pooling
* Adaptive average pooling
* Fully connected regression head
* Dropout: `0.3`
* RUL regression output: `1`

The convolutional baseline is designed to provide a simple local-pattern model for comparison with recurrent and attention-based models.

\---

### 2\. LSTM Baseline

The LSTM baseline is designed to model the temporal evolution of engine degradation directly.

The implementation follows the baseline configuration used in the referenced work:

* Input: `Xs + W`
* Input dimension: `18`
* Window size: `50`
* 3-layer LSTM
* Hidden size: `20`
* Fully connected layer: `50`
* ReLU regression head
* Xavier initialization
* Batch size: `1024`
* Learning rate: `0.001`
* Mini-batch optimization with AMSGrad
* Early stopping patience: `5`

The purpose of this model is to establish a classical recurrent baseline before introducing more complex fusion mechanisms.

\---

### 3\. Pure Transformer Baseline

The Transformer baseline is a **pure temporal self-attention model**. It does not contain CNN branches, physics features, physics losses, or additional fusion modules.

Architecture:

```text
18-D input
   ↓
Linear projection + LayerNorm
   ↓
Sinusoidal positional encoding
   ↓
2-layer Transformer Encoder
   ↓
Last-token representation
   ↓
MLP regression head
   ↓
RUL prediction
```

Main settings:

|Parameter|Value|
|-|-:|
|Input dimension|18|
|Window size|50|
|`d\_model`|32|
|Attention heads|2|
|Encoder layers|2|
|Feed-forward dimension|64|
|Dropout|0.10|
|Batch size|1024|
|Learning rate|`3e-4`|
|Weight decay|`1e-4`|
|Optimizer|AdamW|
|Scheduler|ReduceLROnPlateau|
|Max epochs|80|
|Early stopping patience|12|
|Gradient clipping|1.0|
|Trainable parameters|18,913|

This model is intentionally kept small so that it functions as a clean Transformer baseline rather than an optimized final architecture.

\---

## Transformer baseline result

The current trained Transformer checkpoint was evaluated on the DS02-006 test set.

|Metric|Result|
|-|-:|
|Best validation MSE|**83.582660**|
|Test MSE|**43.135742**|
|Test RMSE|**6.567780**|
|Test MAE|**5.084003**|
|Test windows|**1,253,596**|
|Parameters|**18,913**|

Per-engine results:

|Engine|RMSE|MAE|
|-|-:|-:|
|Unit 11|7.986706|6.610467|
|Unit 14|6.453649|5.091967|
|Unit 15|3.472916|2.744532|

CNN and LSTM results should be added to this table after their final test checkpoints are fixed and evaluated under the same protocol.

\---

## Repository structure

A clean repository can be organized as follows:

```text
N-CMAPSS-DS02-006-Baselines/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md              # Dataset download / placement instructions
│
├── baselines/
│   ├── cnn\_baseline.ipynb
│   ├── lstm\_baseline.ipynb
│   └── transformer\_baseline.py
│
├── checkpoints/
│   ├── best\_cnn\_model.pth
│   ├── best\_lstm\_model.pth
│   └── best\_transformer\_baseline.ptThe exact filenames can be adjusted to match the files committed to the repository.

\---

## Installation

Create a clean Python environment instead of uploading the local `venv` directory.

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\\Scripts\\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Recommended core dependencies:

```text
numpy
h5py
matplotlib
torch
tqdm
jupyter
```

The exact PyTorch installation should be selected according to the local CUDA version and GPU configuration.

\---

## Data placement

The N-CMAPSS `.h5` dataset is intentionally **not included in this repository** because of its large file size.

Place the dataset in the project directory, for example:

```text
N-CMAPSS-DS02-006-Baselines/
├── N-CMAPSS\_DS02-006.h5
├── baselines/
└── results/
```

The baseline scripts expect:

```python
FILE\_PATH = "N-CMAPSS\_DS02-006.h5"
```

\---

## Reproducibility

The experiments use a fixed random seed of `42` where applicable. The data split is performed by engine unit, and normalization is based on the training data rather than the full dataset.

For a fair comparison, the CNN, LSTM, and Transformer baselines should use the same:

* dataset version
* input variables
* engine split
* window definition
* target definition
* normalization strategy
* evaluation metrics

Only the model architecture and its necessary optimization settings should differ.

\---

## Evaluation metrics

The main regression metrics are:

**RMSE**

$$
RMSE = \\sqrt{\\frac{1}{N}\\sum\_{i=1}^{N}(\\hat{y}\_i-y\_i)^2}
$$

**MAE**

$$
MAE = \\frac{1}{N}\\sum\_{i=1}^{N}|\\hat{y}\_i-y\_i|
$$

Lower values indicate better RUL prediction accuracy.

\---

## Scope of the baselines

These models are intended as **baseline references**, not as the final proposed model.

In particular, the baseline stage does not include:

* physics-informed loss
* health-index constraints
* monotonic calibration
* physics-based features
* explicit multi-branch feature fusion
* gated fusion
* cross-attention fusion
* uncertainty modeling

These mechanisms can be introduced later and evaluated against the baseline results.

\---

## References

1. Arias Chao, M., Kulkarni, C., Goebel, K., \& Fink, O. (2021). *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics*. Data, 6(1), 5. https://doi.org/10.3390/data6010005
2. Arias Chao, M., Kulkarni, C., Goebel, K., \& Fink, O. (2022). *Fusing physics-based and deep learning models for prognostics*. Reliability Engineering \& System Safety, 217, 107961. https://doi.org/10.1016/j.ress.2021.107961

\---

## Notes

This repository focuses on establishing a reproducible baseline for N-CMAPSS DS02-006 before moving to improved fusion architectures. Checkpoints, plots, and prediction files should be kept separate from source code so that the repository remains easy to inspect and reproduce.

