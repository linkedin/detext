![Python application](https://github.com/linkedin/detext/workflows/Python%20application/badge.svg)  ![tensorflow](https://img.shields.io/badge/tensorflow-1.14.0-green.svg) ![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)

DeText: A Deep Neural Ranking Framework with Text Understanding
========

![](thumbnail_DeText.png) 

**Relax like a sloth, let DeText do the understanding for you**

## What is it
**DeText** is a **De**ep **Text** understanding framework for NLP related ranking, classification, and language generation tasks.  It leverages semantic matching using deep neural networks to 
understand member intents in search and recommender systems. 
As a general NLP framework, currently DeText can be applied to many tasks, 
including search & recommendation ranking, multi-class classification and query understanding tasks.

## Highlight
Design principles for DeText framework:
* Natural language understanding powered by state-of-the-art deep neural networks
  * Automatic feature extraction with deep models
  * End-to-end training
  * Interaction modeling between ranking sources and targets
* A general framework with great flexibility to meet requirement of different production applications.
  * Flexible deep model types
  * Multiple loss function choices
  * User defined source/target fields
  * Configurable network structure (layer sizes and #layers)
  * Tunable hyperparameters
...
  
* Reaching a good balance between effectiveness and efficiency to meet the industry requirements.

## The framework
The DeText framework contains multiple components:

**Word embedding layer**.  It converts the sequence of words into a d by n matrix.

**CNN/BERT/LSTM for text embedding layer**.  It takes into the word embedding matrix as input, and maps the text data into a fixed length embedding.  It is worth noting that we adopt the representation based methods over the interaction based methods.  The main reason is the computational complexity: The time complexity of interaction based methods is at least O(mnd), which is one order higher than the representation based methods max(O(md), O(nd).

**Interaction layer**.  It generates deep features based on the text embeddings.  Many options are provided, such as concatenation, cosine similarity, etc.

**Traditional Feature Processing**.  We combine the traditional features with the interaction features (deep features) in a wide & deep fashion.

**MLP layer**. The MLP layer is to combine traditional features and deep features. 

It is an end-to-end model where all the parameters are jointly updated to optimize the click probability.

![](detext_model_architecture.png) 

## Model Flexibility
DeText is a general ranking framework that offers great flexibility for clients to build customized networks for their own use cases:

**LTR/classification layer**: in-house LTR loss implementation, or tf-ranking LTR loss, multi-class classification support.

**MLP layer**: customizable number of layers and number of dimensions.

**Interaction layer**: support Cosine Similarity, Outer Product, Hadamard Product, and Concatenation.

**Text embedding layer**: support CNN, BERT, LSTM-Language-Model with customized parameters on filters, layers, dimensions, etc.

**Continuous feature normalization**: element-wise scaling, value normalization.

**Categorical feature processing**: modeled as entity embedding.

All these can be customized via hyper-parameters in the DeText template. Note that tf-ranking is supported in the DeText framework, i.e., users can choose the LTR loss and metrics defined in DeText.

## How to use it
### Setup dev environment

1. Create & source your virtualenv
1. Run setup for DeText:

```bash
python setup.py develop
```

### Run tests

Run all tests:

```bash
pytest 
```

### Running DeText model training toy example
The main script for running DeText model training is through `src/detext/run_detext.py`. Users could customize the hyperparmeters based on different requirements for specific tasks. Please refer to [TRAINING.md](TRAINING.md) for more details on the training data format and hyperparameter description. 
For a test run on a small sample dataset, please 
checkout the following script.
```$xslt
cd src/detext/resources
bash run_detext.sh
```

### DeText training manual

Users have full control for custom designing DeText models. In the training manual ([TRAINING.md](TRAINING.md)), users can find information about the following:
* Training data format and preparation
* Key parameters to customize and train DeText models
* Detailed information about all DeText training parameters for full customization
## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the BSD 2-CLAUSE LICENSE - see the [LICENSE.md](LICENSE.md) file for details
