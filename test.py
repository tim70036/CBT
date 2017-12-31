from cbtest.dataset import *
from cbtest.utils import *
from cbtest.config import *
from cbtest.baseline.embedding import *
from cbtest.evaluate import Experiment
import random
import pdb, traceback, sys

import argparse

sys.setrecursionlimit(1000000000000)

learner = CBTLearner(batchsize=64, hidden_dim=100, lr=args.lr, encoder=args.encoder, 
            position_encoding=args.PE)

learner.mem_size = 10240
learner.unit_size = 1
learner.sen_maxlen = 512 # query sentence len.
learner.encode_context = learner.encode_context_lexical
learner.encode_query = learner.encode_query_lexical
learner.arch = learner.arch_memnet_lexical


experiment = Experiment('memory-net')
experiment.load_pickle(fprop=learner.fprop,
        bprop=learner.bprop)