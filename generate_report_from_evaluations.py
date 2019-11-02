import pandas as pd
import numpy as np

df = pd.read_pickle(
    'cognitive_package/res/reports/model evaluation/evaluation_results.pkl')
RECALL = 'recall'
SPECIFITY = 'specifity'
PRECISION = 'precision'
FALSE_POS_RATE = 'false_pos_rate'
FALSE_NEG_RATE = 'false_neg_rate'
ACCURACY = 'accuracy'


def extract_metrics():

    keys = [RECALL,
            SPECIFITY,
            PRECISION,
            FALSE_POS_RATE,
            FALSE_NEG_RATE,
            ACCURACY]

    total_metrics = {key: 0 for key in keys}
    count = 0

    metric_df = pd.DataFrame(columns=keys)
    for conf_mat in df['confusion_matrice']:
        count += 1
        tp, fp = conf_mat[1][1], conf_mat[1][0]
        tn, fn = conf_mat[0][0], conf_mat[0][1]

        cur_metric = {key: 0 for key in keys}

        cur_metric[RECALL] = tp / (tp + fn)
        cur_metric[SPECIFITY] = tn / (tn + fp)
        cur_metric[PRECISION] = tp / (tp + fp)
        cur_metric[FALSE_POS_RATE] = fp / (fp + tn)
        cur_metric[FALSE_NEG_RATE] = fn / (fn + tp)
        cur_metric[ACCURACY] = (tp + tn) / (tp + tn + fp + fn)

        row = pd.Series(cur_metric)
        metric_df = metric_df.append(cur_metric, ignore_index=True)

        for k in keys:
            total_metrics[k] += cur_metric[k]

    metric_df.to_csv(
        'cognitive_package/res/reports/model evaluation/metrics_table.csv')


def compute_general_scores():
    metrics_df = pd.read_csv(
        'cognitive_package/res/reports/model evaluation/metrics_table.csv')
    print("\nmetrics mean: \n {}".format(metrics_df.mean(axis=0)))
    print("\nmetrics std: \n {}".format(metrics_df.std(axis=0)))
    evaluation_df = pd.read_pickle(
        'cognitive_package/res/reports/model evaluation/evaluation_results.pkl')

    evaluation_df.drop('confusion_matrice', axis=1)

    print("\nevaluation mean: \n {}".format(evaluation_df.mean(axis=0)))
    print("\nevaluation std: \n {}".format(evaluation_df.std(axis=0)))

extract_metrics()
compute_general_scores()
