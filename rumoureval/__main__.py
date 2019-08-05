"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
import sys
from .classification.sdqc import sdqc
from .classification.veracity_prediction import veracity_predict
from .classification.veracity_prediction import veracity_train
from .scoring.Scorer import Scorer
from .util.data import import_data, import_annotation_data, output_data_by_class
from .util.log import setup_logger


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='RumourEval')
    parser.add_argument('--plot', action='store_true',
                        help='plot confusion matrices')
    parser.add_argument('--fit', action='store_true', help='fit and store the model')
    parser.add_argument('--predict', action='store', dest='predict', help='predict the tweets given')
    parsed_args = parser.parse_args()

    ova_parts = ['charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege', 'ebola-essien', 'prince-toronto', 'putinmissing'] #
    logger = setup_logger(False)
    a_results = []
    b_results = []
    i = 1
    logger.info(ova_parts[i])
    logger.info('#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  ')

    # Import training and evaluation datasets
    train_ovas = [x for x in ova_parts if x != ova_parts[i]]
    print(train_ovas)
    tweets_train = import_data('train', train_ovas)
    if(parsed_args.predict != None):
        tweets_eval = import_data(parsed_args.predict, 'tweets')
        annotationEval = import_annotation_data(parsed_args.predict)
    else:
        tweets_eval = import_data('train', ova_parts[i])
        annotationEval = import_annotation_data('train')

    all_tweets = { }

    for tweet in tweets_train:
        all_tweets[str(tweet['id'])] = tweet
    for tweet in tweets_eval:
        all_tweets[str(tweet['id'])] = tweet

    # Import annotation data for training and evaluation datasets
    annotations = import_annotation_data('train')

    a_annotations = { }
    b_annotations = { }
    for x, y in annotations[0].items():
        a_annotations[x] = y
    for x, y in annotations[1].items():
        b_annotations[x] = y
    for x, y in annotationEval[1].items():
        b_annotations[x] = y


    # Get the root tweets for each dataset for veracity prediction
    root_tweets_train = [x for x in tweets_train if x.is_source]
    root_tweets_eval = [x for x in tweets_eval if x.is_source]

    if parsed_args.fit:
        veracity_train(root_tweets_train, a_annotations, b_annotations, parsed_args.plot, all_tweets)
    else:
        task_b_results = veracity_predict(root_tweets_eval, a_annotations, b_annotations, parsed_args.plot)

        printit = True #True if ova_parts[i] == 'ferguson' else False
        task_b_score = PrintScoreB(root_tweets_eval, b_annotations, task_b_results, printit = printit)
        logger.info('Task B score: ' + str(task_b_score) + ', rmse=' + str((task_b_score[2]/task_b_score[1])))
        logger.info('')

        logger.info("Scores!!!!!!")
        t = task_b_score[0]
        f = task_b_score[1]
        logger.info(('True Predicted: ' + str(t) + ' False Predicted: ' + str(f-t) + ' = ' + str(float(t) / (float(f)))))


def PrintScoreA(tweets_eval, a_annotations, results, printit=False):
    sum = 0
    for tweet in tweets_eval:
        x = str(tweet['id'])
        if(str(a_annotations[x]) == str(results[x])):
            sum = sum + 1
        if(printit):
            print('a- ' + str(x) + ': ' + str(a_annotations[x]) + ' ' + str(results[x]) + ' = '  + str(str(a_annotations[x]) == str(results[x])))
    return sum, len(tweets_eval)

def PrintScoreB(tweets_eval, a_annotations, results, printit=False):
    sum = 0
    rmse = 0
    for tweet in tweets_eval:
        x = str(tweet['id'])
        if(str(a_annotations[x]) == str(results[x][0])):
            sum = sum + 1
        rmse = rmse + results[x][1]
        if(printit):
            print('b- ' + str(x) + ': ' + str(a_annotations[x]) + ' ' + str(results[x]) + ' = '  + str(str(a_annotations[x]) == str(results[x][0])))
    return sum, len(tweets_eval), rmse

if __name__ == "__main__":
    main()
