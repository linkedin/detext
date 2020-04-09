#####This is a sample output example that's created through avro2tf.
Output structure on HDFS is:
- *featureList/*
- *metadata/tensor_metadata.json*
   (a generated metadata file that defines the shape of features and labels)
- *trainingData/*.avro* (training data)
- *validationData/*.avro* (dev data)

Here we include a `test.avro` file that has the same format as used for training and the corresponding
 `tensor_metadata.json`. These are required for creating a dataset using `TensorFlowInDatasetUtils` from `from linkedin.tensorflowtraining.io`.

Please see `test_data_fn.py` for usage.