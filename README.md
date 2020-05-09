# Signet-Pytorch

SigNet implementation in Pytorch

Original paper: https://arxiv.org/pdf/1707.02131.pdf

**Note**: I'm not the author of the paper. I'm just curious about the network and topic that I want to reimplement myself in Pytorch

# Download dataset
```bash
cd data
./cedar.sh
```

# Prepare data
This is used to split dataset to train/test partitions

```bash
python3 prepare_data.py
```

# Train
```bash
python3 train.py
```

# Result

| Dataset | Accuracy (%) |
| ------- | ------------ |
| CEDAR | 100 |

# TODO
- [ ] Init bias = 0 as paper proposed
- [ ] Add normalization (e.g. divide by standard deviation)
- [ ] GPDS 300 dataset
- [ ] GPDS Synthetic Signature Corpus dataset
- [ ] Bengali dataset
- [ ] Hindi dataset

# Reference
- (Maybe author's code) Keras implementation https://github.com/sounakdey/SigNet