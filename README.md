# jTrans
This repo is the official code of **jTrans: Jump-Aware Transformer for Binary Code Similarity Detection**. 

![Illustrating the performance of the proposed jTrans](/figures/poolsizecompare.png)

## News
* \[2022/7/7\] We update BinaryCorp with the original [binaries](https://cloud.vul337.team:8443/s/W57ZWXxn7zSKG4q).
* \[2022/6/18\] We release the code and models of jTrans. 
* \[2022/6/9\] We release the preprocessing code and [BinaryCorp](https://cloud.vul337.team:8443/s/cxnH8DfZTADLKCs), the dataset we used in our paper.
* \[2022/5/26\] jTrans is now on [ArXiv](https://arxiv.org/pdf/2205.12713.pdf).

## Get Started
### Prerequisites
- Linux (MacOS and Windows are not currently officially supported)
- Python 3.8+
- PyTorch 1.10+
- CUDA 10.2+
- IDA pro 7.5+ (only used for dataset processing)

### Quick Start

**a. Create a conda virtual environment and activate it.**
```
conda create -n jtrans python=3.8 pandas tqdm -y
conda activate jtrans
```

**b. Install PyTorch and other packages.**
```
conda install pytorch cudatoolkit=11.0 -c pytorch
python -m pip install simpletransformers networkx pyelftools
```

**c. Get code and models of jTrans.**
```
git clone https://github.com/vul337/jTrans.git && cd jTrans
```
Download [experiments.tar.gz](https://cloud.vul337.team:8443/s/wmqzYFyJnSEfEgm) and [models.tar.gz](https://cloud.vul337.team:8443/s/tM5qGQPJa6iynCf) and extract them.
```
tar -xzvf experiments.tar.gz && tar -xzvf models.tar.gz
```

**d. Get the BinaryCorp dataset
Download the processed dataset from this [link](https://cloud.vul337.team:8443/s/cxnH8DfZTADLKCs)**

**e. Finetune new models on the BinaryCorp**
```
python finetune.py -h
```

**d. Evaluation**
```
python eval_save.py -h
python fasteval.py -h
```
try to evaluate jTrans on BinaryCorp-3M after extracting experiments.tar.gz
```
python fasteval.py
```

**f. Try jTrans on your own binaries**

Make sure you have IDA pro 7.5+ and following the instructions at [datautils](datautils/README.md). After extracting features of your binaries, you can try jTrans on them such as the usage at [eval_save.py](./eval_save.py).

## Dataset
- We present a new large-scale and diversified dataset, [BinaryCorp](https://cloud.vul337.team:8443/s/cxnH8DfZTADLKCs), for the task of binary code similarity detection. 
- The description of the dataset can be found at [here](datautils/README.md) and we give an [example](datautils/playdata.py) for using BinaryCorp.
- If you need to use features that we do not provide in advance, such as call graphs, you can download the raw binaries from [here](https://cloud.vul337.team:8443/s/W57ZWXxn7zSKG4q).

## Acknowledgement
This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

* [transformers](https://github.com/huggingface/transformers)
* [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)

## Bibtex
If this work or BinaryCorp dataset are helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2022jtrans,
  title={jTrans: Jump-Aware Transformer for Binary Code Similarity},
  author={Wang, Hao and Qu, Wenjie and Katz, Gilad and Zhu, Wenyu and Gao, Zeyu and Qiu, Han and Zhuge, Jianwei and Zhang, Chao},
  journal={arXiv preprint arXiv:2205.12713},
  year={2022}
}
```