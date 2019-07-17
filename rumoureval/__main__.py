"""RumourEval: Determining rumour veracity and support for rumours."""

import argparse
import sys
from .classification.sdqc import sdqc
from .classification.veracity_prediction import veracity_prediction
from .scoring.Scorer import Scorer
from .util.data import import_data, import_annotation_data, output_data_by_class
from .util.log import setup_logger


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    ######################
    # Set up Environment #
    ######################
    parser = argparse.ArgumentParser(description='RumourEval')
    parser.add_argument('--test', action='store_true',
                        help='run with test data. defaults to run with dev data')
    parser.add_argument('--trump', action='store_true',
                        help='run with trump data. defaults to run with dev data. overridden by --test')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose logging')
    parser.add_argument('--osorted', action='store_true',
                        help='output tweets sorted by class')
    parser.add_argument('--disable-cache', action='store_true',
                        help='disable cached classifier')
    parser.add_argument('--plot', action='store_true',
                        help='plot confusion matrices')
    parser.add_argument('--ova', action='store_true',
                        help='awesome test method')
    parsed_args = parser.parse_args()
    eval_datasource = 'test' if parsed_args.test else ('trump' if parsed_args.trump else 'dev')

    ova_parts = ['charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege', 'ebola-essien', 'prince-toronto', 'putinmissing'] #
    ova_count = 1 if not parsed_args.ova else 4
    ## TODO fix error at 'ebola-essien', 'prince-toronto', 'putinmissing' this is wht ova count is 4

    # Setup logger
    logger = setup_logger(parsed_args.verbose)

    ########################
    # Begin classification #
    ########################

    a_results = []
    b_results = []

    for i in range(ova_count):
        logger.info('#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#  ')
        logger.info(i+1)
        logger.info(ova_parts[i])
        # Import training and evaluation datasets
        train_ovas = [x for x in ova_parts if x != ova_parts[i]]
        print(train_ovas)
        tweets_train = import_data('train', train_ovas)
        tweets_eval = import_data('train', ova_parts[i])
        
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


        # Get the root tweets for each dataset for veracity prediction
        root_tweets_train = [x for x in tweets_train if x.is_source]
        root_tweets_eval = [x for x in tweets_eval if x.is_source]

        # Perform sdqc task
        task_a_results = sdqc(tweets_train,
                            tweets_eval,
                            a_annotations,
                            parsed_args.plot,
                            all_tweets)

        # Perform veracity prediction task
        task_b_results = veracity_prediction(root_tweets_train,
                                            root_tweets_eval,
                                            a_annotations,
                                            b_annotations,
                                            parsed_args.plot,
                                            all_tweets)

        # Score tasks and output results
        #task_a_scorer = Scorer('A', eval_datasource)
        #task_a_scorer.score(task_a_results)

        #task_b_scorer = Scorer('B', eval_datasource)
        #task_b_scorer.score(task_b_results)
        task_a_score = PrintScoreA(tweets_eval, a_annotations, task_a_results)
        printit = False #True if ova_parts[i] == 'ferguson' else False
        task_b_score = PrintScoreB(root_tweets_eval, b_annotations, task_b_results, printit = printit)
        logger.info('Task A score: ' + str(task_a_score))
        logger.info('Task B score: ' + str(task_b_score) + ', rmse=' + str((task_b_score[2]/task_b_score[1])))
        a_results.append(task_a_score)
        b_results.append(task_b_score)
        logger.info('')

    logger.info("Scores!!!!!!")

    sum_true = 0
    sum_all = 0
    for i in range(len(a_results)):
        result = a_results[i]
        logger.info('Score TaskA at ' + '{:15}'.format(ova_parts[i]) + ': ' + str(result[0]) + '/' + str(result[1]))
        sum_true = sum_true + result[0]
        sum_all = sum_all + result[1]
    logger.info(sum_true)
    logger.info(sum_all)
    overall_a_score = float(sum_true) / sum_all
    
    sum_true = 0
    sum_all = 0
    sum_rmse = 0
    for i in range(len(b_results)):
        result = b_results[i]
        logger.info('Score TaskB at ' + '{:15}'.format(ova_parts[i]) + ': ' + str(result[0]) + '/' + str(result[1]) + ', rmse=' + str((result[2]/result[1])))
        sum_true = sum_true + result[0]
        sum_all = sum_all + result[1]
        sum_rmse = sum_rmse + result[2]
    overall_b_score = float(sum_true) / sum_all
    overall_rmse_score = float(sum_rmse) / sum_all
    
    logger.info('Total A    score: ' + str(overall_a_score))
    logger.info('Total B    score: ' + str(overall_b_score))
    logger.info('Total Rmse score: ' + str(overall_rmse_score))

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
            print('b- ' + str(x) + ': ' + str(a_annotations[x]) + ' ' + str(results[x]) + ' = '  + str(str(a_annotations[x]) == str(results[x])))
    return sum, len(tweets_eval), rmse

if __name__ == "__main__":
    main()
