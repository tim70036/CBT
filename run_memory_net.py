from cbtest.dataset import *
from cbtest.utils import *
from cbtest.config import *
from cbtest.baseline.embedding import *
from cbtest.evaluate import Experiment
import random
import pdb, traceback, sys

import argparse

sys.setrecursionlimit(1000000000)

# Parsing Argument
parser = argparse.ArgumentParser(description='LSTM Embedding Baseline')
parser.add_argument('--task', type=str, default='cn')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--iter', type=int, default=20)
parser.add_argument('--encoder', type=str, default='bow')
parser.add_argument('--memory', type=str, default='lexical')
parser.add_argument('--small', action='store_true')
parser.add_argument('--PE', action='store_true')
parser.add_argument('--window_b', type=int, default=2)

args = parser.parse_args()
print colorize('[arguments]\t' + str(args), 'red')

# The default task is 'cn', can be changed by providing argument --task
task = args.task
print '[running task]', task

# The path is init in config.py
train_path = globals()['cbt_' + task + '_train'] # globals() returns a dictionary of all global var
test_path = globals()['cbt_' + task + '_test']
print '[train_path]', train_path
print '[test_path]', test_path

try:

    # Reading Data, dataset.py
    if args.small: # if given --small in arg, then limit 1000 set
        train_exs = read_cbt(train_path, limit=1000)
    else:
        train_exs = read_cbt(train_path)
    test_exs = read_cbt(test_path)

    # Preprocess Data, embedding.py
    learner = CBTLearner(batchsize=64, hidden_dim=100, lr=args.lr, encoder=args.encoder, 
            position_encoding=args.PE)
    # create vocab, a dict that contains all words appear in train data(non repeat)
    learner.create_vocab(train_exs)
    # check all word, if a word is not in vocab, convert it into '<unk>'
    learner.preprocess_dataset(train_exs)
    learner.preprocess_dataset(test_exs)

    # Set up var in learner, assign different function according to the arguments
    if args.memory == 'lexical':
        learner.mem_size = 10240
        learner.unit_size = 1
        learner.sen_maxlen = 512 # query sentence len.
        learner.encode_context = learner.encode_context_lexical
        learner.encode_query = learner.encode_query_lexical
        learner.arch = learner.arch_memnet_lexical
    elif args.memory == 'window':
        param_b = args.window_b
        learner.mem_size = 128
        learner.unit_size = 2 * param_b + 1
        learner.sen_maxlen = 2 * param_b + 1
        learner.encode_context = lambda ex: learner.encode_context_window(ex, param_b=param_b)
        learner.encode_query = lambda ex: learner.encode_query_window(ex, param_b=param_b)
        learner.arch = learner.arch_memnet_lexical
    elif args.memory == 'lstmq':
        param_b = args.window_b
        learner.mem_size = 1024
        learner.unit_size = 2 * param_b + 1
        learner.sen_maxlen = 2 * param_b + 1
        learner.encode_context = lambda ex: learner.encode_context_window(ex, param_b=param_b)
        learner.encode_query = lambda ex: learner.encode_query_window(ex, param_b=param_b)
        learner.arch = lambda: learner.arch_lstmq(param_b=param_b)
    elif args.memory == 'sentence':
        learner.mem_size = 20
        learner.unit_size = 1024
        learner.sen_maxlen = 1024
        learner.encode_context = learner.encode_context_sentence
        learner.encode_query = learner.encode_query_sentence
        learner.arch = learner.arch_memnet_lexical
    elif args.memory == 'selfsup':
        param_b = 2
        learner.mem_size = 1024
        learner.unit_size = 2 * param_b + 1
        learner.sen_maxlen = 2 * param_b + 1
        learner.encode_context = learner.encode_context_selfsup
        learner.encode_query = learner.encode_query_selfsup
        learner.encode_label = learner.encode_label_selfsup
        learner.encode_candidate = learner.encode_candidate_selfsup
        learner.arch = learner.arch_memnet_selfsup
        learner.loss = learner.loss_selfsup
        learner.test = learner.test_selfsup

    # RUNRUNRUN
    learner.compile()

    experiment = Experiment('memory-net-%s' % args.encoder)

    for it in range(args.iter):
        learner.train(train_exs, num_iter=1)
        (acc, errs) = learner.test(test_exs)
        print '[epoch %d]' % it, 'accuracy = ', acc
        experiment.log_json(result={
            'task': task,
            'acc': acc,
            'errs': errs
        })

        # Save model every 5 rounds, evaluate.py
        if it % 5 == 0:
            print 'saving model...'
            print colorize('[arguments]\t' + str(args), 'red')
            experiment.log_pickle(fprop=learner.fprop,
            bprop=learner.bprop)

        experiment.next()
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

print 'saving model...'
# Save model, evaluate.py
experiment.log_pickle(fprop=learner.fprop,
        bprop=learner.bprop)
