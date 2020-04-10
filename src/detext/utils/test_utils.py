"""
A lot of non-tensorflow implementation.  This is mainly used for testing.
"""
import argparse
import math
import numpy as np


def neg_log_sigmoid(x):
    """Negative log sigmoid"""
    return -math.log(1 / (1 + math.exp(-x)))


def compute_dcg_power2(y_score, y_true, topk):
    """Computes dcg with relevance score power(2, y_true)"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:topk])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def compute_ndcg_power2(y_score, y_true, topk):
    """Computes idcg with relevance score power(2, y_true)"""
    dcg = compute_dcg_power2(y_score, y_true, topk)
    print(dcg)
    idcg = compute_dcg_power2(y_true, y_true, topk)
    print(idcg)
    return dcg / idcg if idcg > 0 else 0


def get_lambda_loss(scores, labels, group_size, topk=None):
    """A non tf implementation of lambda loss with ndcg"""
    batch_size = len(scores)
    max_group_size_scores = len(scores[0])
    loss = []
    for i in range(batch_size):
        loss.append([])
        for j in range(max_group_size_scores):
            loss[i].append([])
            for k in range(max_group_size_scores):
                if j < group_size[i] and k < group_size[i] and labels[i][j] > labels[i][k]:
                    loss[i][j].append(neg_log_sigmoid(scores[i][j] - scores[i][k]))
                    if topk:  # lambda rank
                        ndcg1 = get_ndcg(scores[i][:group_size[i]], labels[i][:group_size[i]], topk)
                        new_score = list(scores[i])
                        new_score[j] = scores[i][k]
                        new_score[k] = scores[i][j]
                        ndcg2 = get_ndcg(new_score[:group_size[i]], labels[i][:group_size[i]], topk)
                        ndcg_diff = abs(ndcg1 - ndcg2)
                        loss[i][j][k] *= ndcg_diff
                else:
                    loss[i][j].append(0)
    return loss


def get_ndcg(scores, labels, topk):
    """compute ndcg"""
    # idcg
    idcg, k = 0, 0
    for i, s in enumerate(sorted(labels, reverse=True)):
        idcg += s / math.log(2 + i, 2)
        k += 1
        if k >= topk:
            break
    # dcg
    dcg, k = 0, 0
    for i, (j, _) in enumerate(sorted(enumerate(scores), key=lambda x: x[1], reverse=True)):
        dcg += labels[j] / math.log(2 + i, 2)
        k += 1
        if k >= topk:
            break
    return dcg / idcg


def get_softmax_loss(scores, labels, group_size):
    loss = []
    for score, label, group_s in zip(scores, labels, group_size):
        # find the index with largest value in label
        sm_score = []
        for i in range(len(score)):
            if i >= group_s:
                exp_score = 0
            else:
                exp_score = math.exp(score[i])
            sm_score.append(exp_score)
        sm_score_sum = sum(sm_score)

        for i in range(min(len(score), group_s)):
            sm_score[i] = -math.log(sm_score[i] / sm_score_sum) * label[i]
        loss.append(sm_score)
    return loss


def compute_ndcg_elem_ps_parity_check(result_list, label_index, top_n):
    dcg = 0
    for i, result in enumerate(result_list):
        if i >= top_n:
            break
        index = i + 1
        score = float(result[label_index])
        dcg = dcg + score / math.log(index + 1, 2)
    return dcg


def compute_ndcg_ps_parity_check_original_api(input_list, n):
    # compute dcg
    # dcg = \sum_i {label_i / log_2(pos_i + 1)}, i ordered by prediction score
    result_order_by_score = sorted(input_list, key=lambda f: f[2], reverse=True)
    dcg = compute_ndcg_elem_ps_parity_check(result_order_by_score, 1, n)

    # compute idcg
    # idcg = \sum_i {label_i / log_2(pos_i + 1)}, i ordered by relevance score (click)
    result_order_by_click = sorted(input_list, key=lambda f: f[1], reverse=True)
    idcg = compute_ndcg_elem_ps_parity_check(result_order_by_click, 1, n)

    # calculate ndcg
    ndcg = 0.0
    if idcg > 0.0:
        ndcg = dcg / idcg
    return ndcg


def compute_ndcg_ps_parity_check(scores, labels, topk=10):
    """
    adapter for pps ndcg calculating util
    :param scores: raw scores
    :param labels: actual labels, ordered numerically
    :param topk: default is 10
    :return: ndcg score as float
    """
    return compute_ndcg_ps_parity_check_original_api(list(zip(scores, labels, scores)), topk)


def add_arguments(parser):
    """Build ArgumentParser."""

    # network
    parser.add_argument("--model_dir", type=str, help="exported model dir")
    parser.add_argument("--test_files", type=str, help="test files to be evaluated on")
    parser.add_argument("--doc_text_fields", type=str, help="document text fields")
    parser.add_argument("--usr_text_fields", type=str, help="user text fields")
    parser.add_argument("--num_wide", type=int, help="number of wide features per doc")
    parser.add_argument("--output_dir", type=str, help="result file written in output dir")


def get_params(argv):
    """
    Get hyper-parameters.
    """
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    hparams, unknown_params = parser.parse_known_args(argv)

    # Print all hyper-parameters
    for k, v in sorted(vars(hparams).items()):
        print('--' + k + '=' + str(v))
    return hparams
