"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
from classification.sdqc import sdqc
from classification.veracity_prediction import veracity_prediction
from scoring.Scorer import Scorer
from util.data import import_data, import_annotation_data
from util.log import setup_logger


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='RumourEval, by Tong Liu and Joseph Roque')
    parser.add_argument('--test', action='store_true',
                        help='run with test data. defaults to run with dev data')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')
    return parser.parse_args()


def main(args):
    """Execute RumourEval program."""
    test_datasource = 'test' if args.test else 'dev'
    tweets_train = import_data('train')
    tweets_test = import_data(test_datasource)
    train_annotations = import_annotation_data('train')
    test_annotations = import_annotation_data(test_datasource)

    root_tweets_train = [x for x in tweets_train if x.is_source]
    root_tweets_test = [x for x in tweets_test if x.is_source]

    task_a_results = sdqc(tweets_train, tweets_test, train_annotations[0], test_annotations[0])
    task_b_results = veracity_prediction(root_tweets_train, root_tweets_test, train_annotations[1], test_annotations[1])

    task_a_scorer = Scorer('A', test_datasource)
    task_a_scorer.score(task_a_results)

    task_b_scorer = Scorer('B', test_datasource)
    task_b_scorer.score(task_b_results)


if __name__ == "__main__":
    ARGS = parse_args()
    setup_logger(ARGS.verbose)
    main(ARGS)
