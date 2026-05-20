# LoRa RFFI PyTorch Examples

This work reproduces the research presented in the paper 'Towards Scalable and Channel-Robust Radio Frequency Fingerprint Identification for LoRa', available at the link:(https://ieeexplore.ieee.org/abstract/document/9715147)

This repository contains two PyTorch implementations for LoRa radio frequency
fingerprint identification (RFFI):

- `Closed_set RFFI`: closed-set device classification.
- `Openset_RFFI`: open-set RFFI with triplet-loss feature extraction, KNN
  classification, and rogue-device detection.

The dataset and pretrained extractor weights are expected under `./LoRa_RFF`.
Run commands from this repository root so the relative paths resolve correctly.

## Repository Layout

```text
.
├── Closed_set RFFI/
│   ├── main.py
│   ├── dataset_preparation.py
│   ├── deep_learning_models.py
│   └── training_utils.py
├── Openset_RFFI/
│   ├── main.py
│   ├── dataset_preparation.py
│   └── deep_learning_models.py
└── LoRa_RFF/
    ├── dataset/
    │   ├── Train/
    │   └── Test/
    └── models/
```

Example dataset paths:

```text
./LoRa_RFF/dataset/Test/dataset_residential.h5
./LoRa_RFF/dataset/Test/channel_problem/A.h5
./LoRa_RFF/dataset/Train/dataset_training_aug.h5
./LoRa_RFF/models/Extractor_1.h5
```

## Environment

Install the common dependencies:

```bash
pip install numpy scipy h5py torch scikit-learn matplotlib seaborn
```

CUDA is optional. Both programs automatically fall back to CPU when CUDA is not
available.

## Open-Set RFFI

Run the default classification demo:

```bash
python ./Openset_RFFI/main.py
```

This uses:

- enrollment data: `./LoRa_RFF/dataset/Test/dataset_residential.h5`
- test data: `./LoRa_RFF/dataset/Test/channel_problem/A.h5`
- pretrained extractor: `./LoRa_RFF/models/Extractor_1.h5`

Run rogue-device detection:

```bash
python ./Openset_RFFI/main.py --task rogue
```

Convert an original Keras extractor to a native PyTorch checkpoint:

```bash
python ./Openset_RFFI/main.py --task convert \
  --model ./LoRa_RFF/models/Extractor_1.h5 \
  --output-model ./Openset_RFFI/models/Extractor_1.pth
```

Train a new open-set feature extractor:

```bash
python ./Openset_RFFI/main.py --task train \
  --train-file ./LoRa_RFF/dataset/Train/dataset_training_aug.h5 \
  --output-model ./Openset_RFFI/models/Extractor.pth
```

Quick smoke test:

```bash
python ./Openset_RFFI/main.py --task smoke-test --device cpu
```

## Closed-Set RFFI

Run the default channel experiment:

```bash
python "./Closed_set RFFI/main.py"
```

This trains on `./LoRa_RFF/dataset/Test/channel_problem/A.h5` and evaluates on
`B.h5` through `F.h5`.

Run a short smoke test:

```bash
python "./Closed_set RFFI/main.py" --mode smoke --device cpu
```

Train only:

```bash
python "./Closed_set RFFI/main.py" --mode train \
  --train-file ./LoRa_RFF/dataset/Test/channel_problem/A.h5 \
  --model-out ./Closed_set_RFFI_cnn.pth
```

Evaluate a saved checkpoint:

```bash
python "./Closed_set RFFI/main.py" --mode test \
  --checkpoint ./Closed_set_RFFI_cnn.pth \
  --test-files ./LoRa_RFF/dataset/Test/channel_problem/B.h5
```

## Download Dataset

Please downlaod the dataset and put it in the project folder. The download link is https://ieee-dataport.org/open-access/lorarffidataset.

## Label Convention

HDF5 labels are converted to zero-based device IDs after loading. For example,
devices stored as `31..40` in the HDF5 files are represented as `30..39` inside
the code. Command-line device ranges use these zero-based IDs.
