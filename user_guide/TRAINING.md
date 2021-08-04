DeText Training Manual
==========

## Training data format and preparation

DeText uses TFRecords format for training data.  In general, the input data should have:
* One field for "query" with name `query`
* One field for "wide features" with name `wide_ftrs`
* Multiple fields for "document fields" with name `doc_<field name>`
* One field for "labels" with name `label`
* [optional] Mutiple fields for "user fields" with name `usr_<field name>`
* [optional] One field for "sparse wide features indices" with name `wide_ftrs_sp_idx` and one field for 
  "sparse wide features values" with name `wide_ftrs_sp_val`

We show an example of the prepared training data and explain the data format and shapes. 
* `query` (string list containing only 1 string)
    * For each training sample, there should be 1 query field.
    * eg. ["how do you del ##ete messages"]
* `wide_ftrs` (float list) 
    * There could be multiple dense wide features for each document. Therefore the dense wide features are a 2-D array 
    with shape [#documents, #dense wide features per document]. Since TFRecords support 1-D FloatList, we flatten the 
    dense wide features in the preparation and transform to grouped features in `train/data_fn.py` by reshaping. 
    Therefore the float list of wide_ftrs in the training data has `#documents * #dense wide features per document = 4 * 3 = 12` entries. 
    The dense wide features belong to each document sequentially. I.e., the first 3 wide features belong to the first 
    document, the second 3 wide features belong to the second document, etc..
    * [0.305 0.264 0.180 0.192 0.136 0.027 0.273 0.273 0.377 0.233 0.264 0.227]
* `wide_ftrs_sp_idx` (int list) 
    * There could be multiple sparse wide features for each document. Therefore the sparse wide features indices are a 
    2-D array with shape [#documents, #max num of sparse wide features among documents]. Since TFRecords support 1-D 
    IntList, we flatten the sparse wide features indices in the preparation and transform to grouped features in 
    `train/data_fn.py` by reshaping. Therefore the int list of wide_ftrs_sp_idx in the training data has 
    `#sum_i(num documents in list i * #max sparse wide features in list i)` entries. 
    Within the same list, if the number 
    of sparse feature of document m is smaller than max number of sparse wide features in the list, the sparse feature 
    indices must be padded with 0. An example below shows the wide_ftrs_sp_idx for a list where the maximum number 
    of sparse wide features is 2 and the list has 4 documents. The sparse wide features belong to each document 
    sequentially. I.e., the first 2 wide features belong to the first document, the second 2 wide features belong to 
    the second document, etc.. 
    Note that **0 should NEVER be used for wide_ftrs_sp_idx except for padding**.
    * [3 2 5000 20 1 0 8 0]
* `wide_ftrs_sp_val` (float list) 
   * Sparse wide feature values are in the same shape and must be correspondent to as sparse wide feature indices. I.e., 
   if the sparse feature indices of list i is [1, 5, 2], then the sparse feature values [-5.0, 12.0, 11.0] means that 
   the sparse wide features for this list is [-5.0, 11.0, 0.0, 0.0, 12.0]. If this field is missing, values 
   corresponding to sparse wide feature indices will be set to 1 by default. Values corresponding to padding values of 
   sparse wide feature indices must be set to 0.
   * [3 2 5000 20 1 0 8 0] 
* `label` (float list)
    * The labels corresponding to each document. In our example, 0 for documents without any click and 1 for documents with clicks.
    * [0 0 1 0 0 0 0 0 0 0]
* `doc_titles` (string list)
    * Document text fields. The shape should be the same as label. There could be multiple doc_fields in the data. For example, we could also include a doc_description as a feature. If multiple doc_fields are present, the interaction features will be computed for each query-doc pair. 
    * ["creating a linked ##in group", "edit your profile"...]


## Customizing and training a DeText model

The following example (from [run_detext.sh](src/detext/resources/run_detext.sh)) shows how you could train a DeText CNN model for a search ranking task. 

The train/dev/test datasets are prepared in the format mentioned in the previous section. More specifically, the following fields are used:
* `query`
* `wide_ftrs`
* `wide_ftrs_sp_idx`
* `wide_ftrs_sp_val`
* `doc_titles`
* `label`

The DeText model will extract deep features using the CNN module from both `query` and `doc_titles`. After the text representation, cosine similarity interaction feature between the two fields is computed. The interaction score is then concatenated with the `wide_ftrs`. A dense hidden layer is added before computing the final LTR score.

The following script is used for running the DeText training. 
```bash
python run_detext.py \
--ftr_ext=cnn \ # deep module is CNN
--feature_names=query,label,wide_ftrs,doc_title \   # list all the feature names in the data
--learning_rate=0.001 \
--ltr=softmax \ # type of ltr loss
--max_len=32 \  # sentence max length
--min_len=3 \   # sentence min length
--num_fields=1 \    # the number of document fields (starting with 'doc_') used
--filter_window_sizes=2,3 \   # CNN filter sizes. Could be a list of different sizes.
--num_filters=50 \  # number of filters in CNN
--num_hidden=100 \  # size of hidden layer after the interaction layer
--num_train_steps=10 \
--num_units=32 \    # word embedding size
--num_wide=10 \     # number of wide features per document
--optimizer=bert_adam \
--pmetric=ndcg@10 \     primary metric. This is used for evaluation during training. Best models are kept according to this metric.
--random_seed=11 \
--steps_per_stats=1 \
--steps_per_eval=2 \
--test_batch_size=2 \
--train_batch_size=2 \
--use_wide=True \   # whether to use wide_ftrs
--use_deep=True \   # whether to use the text features
--dev_file=hc_examples.tfrecord \
--test_file=hc_examples.tfrecord \
--train_file=hc_examples.tfrecord \
--vocab_file=vocab.txt \
--out_dir=detext-output/hc_cnn_f50_u32_h100 \
```

The primary parameters are included with comments. Please also find the complete list of training parameters in the next section.

# List of all DeText parameters

A complete list of training parameters that DeText provides is given in [args.py](src/detext/args.py). Attributes of 
the Arg class are accepted arguments. Trailing comments following the attributes are the instructions for using the
argument. E.g.,
```python
@dataclass
class DatasetArg(Arg):
    """Dataset related arguments"""
    distribution_strategy: str = ''  # Distributed training strategy. Reference: tf official models: official/common/distribute_utils.py#L102
    ...
```
means that there's an argument named "distribution_strategy" accepting string format value. This argument stands for
distributed training strategy and users can reference tf official models code to find more instructions (as stated
in the comment).

Supported configurations span across dataset (DatasetArg), feature (FeatureArg), neural network (NetworkArg) and 
optimization (OptimizationArg). Users can check the argument class to customize model training and the model 
architecture.