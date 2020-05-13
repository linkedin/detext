"""
Reference: https://github.com/bluecamel/best_checkpoint_copier
"""
import os
import tensorflow as tf


class Checkpoint(object):
    dir = None
    file = None
    score = None
    path = None

    def __init__(self, path, score):
        self.dir = os.path.dirname(path)
        self.file = os.path.basename(path)
        self.score = score
        self.path = path


class BestCheckpointCopier(tf.estimator.Exporter):
    """
    Copy the best checkpoints for best model evaluation on test after training, and export best savedmodels for serving.
    """
    checkpoints = None
    checkpoints_to_keep = None
    compare_fn = None
    name = None
    pmetric = None
    sort_key_fn = None
    sort_reverse = None

    def __init__(self,
                 serving_input_receiver_fn,
                 name='best_checkpoints',
                 checkpoints_to_keep=1,
                 exports_to_keep=1,
                 pmetric='Loss/total_loss',
                 compare_fn=lambda x, y: x.score < y.score,
                 sort_reverse=False,
                 eval_log_file=None):
        """
        Construct BestCheckpointCopier

        :param serving_input_receiver_fn: serving input function for savedmodel (online serving model).
        :param name: name (dir path) to save the checkpoints and savedmodels.
        The best checkpoints will be copied to ./<name> and the savedmodels are exported to ./export/<name>
        :param checkpoints_to_keep: number of best checkpoints to keep.
        :param exports_to_keep: number of best savedmodels to keep.
        :param pmetric: primary metric to use for determining whether best model is achieved,
        :param compare_fn: compare function to judge current model with previous best.
        :param sort_reverse: sort reverse logic for checking which checkpoints to keep/remove. The top
        <checkpoints_to_keep> checkpoints will be kept. Eg. if the checkpoint with a higher score is considered
        better (compare_fn=lambda x, y: x.score > y.score), then sort_reverse should be set as True.
        """
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.checkpoints = []
        self.checkpoints_to_keep = checkpoints_to_keep
        self._exports_to_keep = exports_to_keep
        self.compare_fn = compare_fn
        self.name = name
        self.pmetric = pmetric
        self.sort_reverse = sort_reverse
        self.eval_log_file = eval_log_file
        self._log("***** Evaluation on dev set during training *****")
        super().__init__()

    def _copyCheckpoint(self, checkpoint):
        """
        Copy the checkpoint.
        """
        desination_dir = self._destinationDir(checkpoint)
        tf.gfile.MkDir(desination_dir)

        for file in tf.gfile.Glob(r'{}*'.format(checkpoint.path)):
            tf.gfile.Copy(file, os.path.join(desination_dir, os.path.basename(file)), overwrite=True)

    def _writeCheckpointStatusFile(self, checkpoint):
        """
        Write to 'checkpoint' with current checkpoint info.
        """
        checkpointStatusFile = os.path.join(self._destinationDir(checkpoint), 'checkpoint')
        with tf.gfile.Open(checkpointStatusFile, 'w') as fout:
            fout.write("model_checkpoint_path: \"{}\"".format(checkpoint.file))

    def _destinationDir(self, checkpoint):
        """
        Best checkpoints directory.
        """
        return os.path.join(checkpoint.dir, self.name)

    def _keepCheckpoint(self, checkpoint):
        """
        Add checkpoint to keep.
        """
        self._log('keeping checkpoint {} with {} = {}'.format(checkpoint.file, self.pmetric, checkpoint.score))
        self.checkpoints.append(checkpoint)
        self.checkpoints = sorted(self.checkpoints, key=lambda x: x.score, reverse=self.sort_reverse)
        self._copyCheckpoint(checkpoint)
        # Write to 'checkpoint' file with best checkpoint info.
        self._writeCheckpointStatusFile(self.checkpoints[0])

    def _log(self, statement):
        """
        Formatting log.
        """
        print(statement)
        if self.eval_log_file is not None:
            with tf.gfile.Open(self.eval_log_file, 'a') as fout:
                fout.write(statement + '\n')

    def _pruneCheckpoints(self, checkpoint):
        """
        Keep top <checkpoints_to_keep> checkpoints.
        """
        destination_dir = self._destinationDir(checkpoint)
        for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
            self._log('removing old checkpoint {} with {} = {}'.format(checkpoint.file, self.pmetric, checkpoint.score))
            old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
            for file in tf.gfile.Glob(r'{}*'.format(old_checkpoint_path)):
                tf.gfile.Remove(file)
        self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

    def _score(self, eval_result):
        """
        Get the score for metric to compare and add logging.
        :param eval_result: current eval result containing the step info and metrics
        :return:
        """
        self._log("## Step {}".format(eval_result.get('global_step', -1)))
        for metric in eval_result:
            if metric != self.pmetric and metric != 'global_step':
                cm = eval_result[metric]
                if not hasattr(cm, "__len__"):
                    self._log("{} : {}".format(metric, cm))
                else:
                    self._log("{} : ".format(cm))
                    self._log(str(cm))
        return float(eval_result[self.pmetric])

    def _shouldKeep(self, checkpoint):
        """
        Whether current checkpoint should be kept.
        """
        return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        """
        Perform best checkpoint copy and best savedmodel export if current checkpoint outperforms previous ones.
        """
        export_result = None
        score = self._score(eval_result)
        checkpoint = Checkpoint(path=checkpoint_path, score=score)
        self._log('Checking checkpoint {}'.format(checkpoint.file))

        if self._shouldKeep(checkpoint):
            self._keepCheckpoint(checkpoint)
            self._pruneCheckpoints(checkpoint)
            export_result = estimator.export_saved_model(
                export_path,
                self.serving_input_receiver_fn,
                checkpoint_path=checkpoint_path)
            tf.estimator.BestExporter._garbage_collect_exports(self, export_path)
        else:
            self._log('skipping checkpoint {} with {} = {}'.format(checkpoint.file, self.pmetric, checkpoint.score))
        # print new line.
        self._log('')
        return export_result
