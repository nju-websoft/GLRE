# Global-to-Local Neural Networks for Document-Level Relation Extraction
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/GLRE/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/GLRE/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)

> Relation extraction (RE) aims to identify the semantic relations between named entities in text. Recent years have witnessed it raised to the document level, which requires complex reasoning with entities and mentions throughout an entire document. In this paper, we propose a novel model to document-level RE, by encoding the document information in terms of entity global and local representations as well as context relation representations. Entity global representations model the semantic information of all entities in the document, entity local representations aggregate the contextual information of multiple mentions of specific entities, and context relation representations encode the topic information of other relations. Experimental results demonstrate that our model achieves superior performance on two public datasets for document-level RE. It is particularly effective in extracting relations between entities of long distance and having multiple mentions.

## Getting Started

### Package Description
```
GLRE/
├─ configs/
    ├── cdr_basebert.yaml: config file for CDR dataset under ``Train'' setting
    ├── cdr_basebert_train+dev.yaml: config file for CDR dataset under ``Train+Dev'' setting
    ├── docred_basebert.yaml: config file for DocRED dataset under ``Train'' setting
├─ data/: raw data and preprocessed data about CDR and DocRED dataset
    ├── CDR/
    ├── DocRED/
├─ data_processing/: the data preprocessing scripts
├─ results/: the pre-trained models and results 
├─ scripts/: the run scripts
├─ src/
    ├── data/: read data and convert to batch
    ├── models/: the core module to implement GLRE
    ├── nnet/: the sub layers to implement GLRE
    ├── utils/: utility function
    ├── main.py:
```

### Dependencies
  - python (>=3.6)
  - pytorch (>=1.5)
  - numpy (>=1.13.3)
  - recordtype (>=1.3)
  - yamlordereddictloader (>=0.4.0)
  - tabulate (>=0.8.7)
  - transformers (>=2.8.0)
  - scipy (>=1.4.1)
  - scikit-learn (>=0.22.1)

### Usage
#### Datasets & Pre-processing
The datasets include CDR and DocRED. The data are located in `data/CDR` directory and `data/DocRED` directory respectively. 
The pre-processing scripts are located in the `data_processing` directory and the pre-processing results are located in the `data/CDR/processed` directory and `data/DocRED/processed` directory respectively.
The pre-trained models are in the `results` directory.

Specifically, we preprocessed the CDR dataset following [Edge-oriented Graph](https://github.com/fenchri/edge-oriented-graph).

    Download the GENIA Tagger and Sentence Splitter:
    $ cd data_processing
    $ mkdir common && cd common
    $ wget http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && tar xvzf geniass-1.00.tar.gz
    $ cd geniass/ && make && cd ..
    $ git clone https://github.com/bornabesic/genia-tagger-py.git
    $ cd genia-tagger-py 
    
    Here, you should modify the Makefile inside genia-tagger-py and replace line 3 with `wget http://www.nactem.ac.uk/GENIA/tagger/geniatagger-3.0.2.tar.gz`
    $ make
    $ cd ../../
    
    In order to process the datasets, they should first be transformed into the PubTator format. The run the processing scripts as follows:
    $ sh process_cdr.sh

And use the following code to preprocess the DocRED dataset.
    
    python docRedProcess.py --input_file ../data/DocRED/train_annotated.json \
                       --output_file ../data/DocRED/processed/train_annotated.data \
    
#### Train & Test
The default hyper-parameters are in the `configs` directory and the train&test scripts are in the `scripts` directory. 
Besides, the `run_cdr_train+dev.py` script corresponds to the CDR under `traing + dev` setting.

    python run_cdr.py
    python run_cdr_train+dev.py
    python run_docred.py

#### Evaluation

For CDR, you can evaluate the result using the evaluation script as follows:


    python utils/evaluate_cdr.py --gold ../data/CDR/processed/test.gold --pred ../results/cdr-dev/cdr_basebert_full/test.preds --label 1:CID:2

For DocRED, you can submit the `result.json` to [Codalab](https://competitions.codalab.org/competitions/20717).

## License

This project is licensed under the GPL License - see the [LICENSE](https://github.com/nju-websoft/GLRE/blob/master/LICENSE) file for details.


## Citation

If you use this work or code, please kindly cite the following paper:

```
@inproceedings{GLRE,
 author = {Difeng Wang and Wei Hu and Ermei Cao and Weijian Sun},
 title = {Global-to-Local Neural Networks for Document-Level Relation Extraction},
 booktitle = {EMNLP},
 year = {2020},
}
```

## Contacts

If you have any questions, please feel free to contact [Difeng Wang](mailto:dfwang.nju@gmail.com), we will reply it as soon as possible.
