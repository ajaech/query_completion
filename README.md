# Personalized Query Completion

This repo contains code for building an LSTM LM for personalized query auto-completion. The model along with experimental results and baselines are described in our 2018 ACL paper, currently available on arXiv. https://arxiv.org/pdf/1804.09661.pdf

Train a model using
`
./trainer.py /path/to/expdir --data /path/to/data.tsv --valdata /path/to/valdata.tsv
`
Set hyperparameters following the format in `default_params.json`.

Description of code files:
* beam.py - helper code for doing beam search
* factorcell.py - implementation of the FactorCell recurrent layer
* model.py - defines the Tensorflow graph for the language model
* trainer.py - script for training a new langauge model
* dynamic.py - script for evaluating trained model on new users

