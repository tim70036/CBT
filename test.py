from cbtest.dataset import *
from cbtest.utils import *
from cbtest.config import *
from cbtest.baseline.embedding import *
from cbtest.evaluate import Experiment
from six.moves import cPickle as pickle
from os import path
import random
import pdb, traceback, sys
import dill
import json
import argparse
import string
import collections
import csv

def write_csv(path, order_dict):
    with open(path,'wb')as csvfile:
        wrtfile = csv.writer(csvfile, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wrtfile.writerow(['id','answer'])
        for k, v in order_dict.iteritems():
            wrtfile.writerow([k,v])
        
def read_list(path):
    file = open(path, 'r')
    file.seek(0)
    new_list = []
    for item in [s.replace('\n','') for s in file.readlines()]:
	if item not in new_list:
	    new_list.append(item)
    file.close()
    return new_list

def write_list(list_obj,text_str):
    file = open('./cand_pool/'+text_str+'.txt', 'w')
    lines=list_obj
    newlines=[str(line)+'\n'  for line in lines]
    file.writelines(newlines)
    file.close()


#read test set without answer
def read_test_set(path, limit=None):
    '''
    read in a children's book dataset.
    return list of dicts. Each dict is an object with keys:
        1. context   # the context sentences.
        2. query     # the query with blank.
        3. answer    # correct answer word.
        4. candidate # list of answer candidates
    '''
    with open(path, 'r') as f:
        exs = []
        context = []
        for line in f:
            line = line.replace('\n', '')

            # empty?
            if line == '':
                continue

            # process 
            m = re.match(r'[0-9]* ', line).end() # get the index of line number, r'' means raw string
            line_no = int(line[:m-1]) #  the line number
            sentence = line[m:] # the sentence

            # if it is query.
            if line_no == 21: 
                sentence = sentence.split('\t') # the ans ,query and cand are seperated by a tab
                query = sentence[0].strip().split(' ')
                candidate = sentence[1].strip().split('|')
                candidate = [c for c in candidate if c] # now candidate is a list of all cand words

                while len(candidate) < 10:
                    candidate.append('<null>')
                assert(len(candidate) == 10)

                # ex is a dict
                ex = {
                    'context': context,
                    'query': query,
                    'candidate': candidate
                }
                assert(len(context) == 20)

                # append to a list of dicts -> exs
                exs.append(ex)

                # if we only want to train small amount, --small provided in arg
                if limit and len(exs) > limit:
                    break
                context = []

            # if it is normal sentence, append to context, it will be add to exs after we meet a query
            else:
                context.append(sentence.strip().split(' '))

        return exs


#preprocess dataset without answer		
def preprocess_test_set(learner, exs):
    def preprocess(sentence):
        sen = unkify(learner.preprocess_sentence(sentence), learner.vocab)
        #if len(sen) > self.sen_maxlen:
        #    print '[warning] exceeding sentence max length.'
        #    sen = sen[:self.sen_maxlen]
        return sen

    # Examine given data(train or test data)
    # Check each word in a sentence(a list of words)
    # If a word is not in vocab, convert it into '<unk>'
    for ex in exs:
        new_context = []
        for sen in ex['context']:
            sen = preprocess(sen)
            new_context.append(sen)
        ex['context'] = new_context
        ex['query'] = preprocess(ex['query'])
        ex['candidate'] = preprocess(ex['candidate'])

# encode_minibathch without answer			
def encode_minibatch_test(learner, minibatch):
    contexts = []
    querys = []

    cvs = []
    for ex in minibatch:
        query = learner.encode_query(ex)
        contexts.append(learner.encode_context(ex))
        querys.append(learner.encode_query(ex))

        cvs.append(learner.encode_candidate(ex))
    contexts = np.vstack([context[np.newaxis, ...] for context in contexts])
    querys = np.array(querys, dtype=np.int64)
    cvs = np.array(cvs, dtype=np.int64)
    return (contexts, querys, cvs)


# build answer list
def pred(learner, exs):
    preprocess_test_set(learner,exs)

    all_preds = []
    cnt = 0 # for remove extra ans	
    for offset in range(0, len(exs), learner.batchsize):
        minibatch = exs[offset:offset + learner.batchsize]
        while len(minibatch) < learner.batchsize:
            minibatch.append(minibatch[-1])
	    cnt += 1
        (contexts, querys, cvs) = encode_minibatch_test(learner,minibatch)

        inds = np.argmax(learner.fprop(contexts, querys, cvs)
                            [np.transpose([range(cvs.shape[0])] * cvs.shape[1]),
                            cvs], axis=1)
        preds = cvs[range(learner.batchsize), inds]
		
        for i in range(0,learner.batchsize):
            if learner.ivocab[preds[i]] in minibatch[i]['candidate']:
                print minibatch[i]['candidate'].index(learner.ivocab[preds[i]])
		all_preds.append(minibatch[i]['candidate'].index(learner.ivocab[preds[i]]))
    all_preds = all_preds[0:len(all_preds)-cnt]
    return all_preds

sys.setrecursionlimit(1000000000)

# Parsing Argument
parser = argparse.ArgumentParser(description='LSTM Embedding Baseline')
parser.add_argument('--task', type=str, default='cn')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dim', type=int, default=200)
parser.add_argument('--iter', type=int, default=20)
parser.add_argument('--encoder', type=str, default='bow')
parser.add_argument('--memory', type=str, default='lexical')
parser.add_argument('--small', action='store_true')
parser.add_argument('--PE', action='store_true')
parser.add_argument('--window_b', type=int, default=2)
parser.add_argument('--model', type=str, default='')

# yes for get candidate_pool file , no for rebuild candidate pool
parser.add_argument('--get_set' , type=str , default='no')
args = parser.parse_args()
print colorize('[arguments]\t' + str(args), 'red')

# The default task is 'cn', can be changed by providing argument --task
task = args.task
print '[running task]', task

# The path is init in config.py
train_cn_path = globals()['cbt_cn_train']
test_cn_path = globals()['cbt_cn_test']

train_ne_path = globals()['cbt_ne_train']
test_ne_path = globals()['cbt_ne_test']

train_p_path = globals()['cbt_p_train']
test_p_path = globals()['cbt_p_test']

train_v_path = globals()['cbt_v_train']
test_v_path = globals()['cbt_v_test']

test_path = globals()['test_set']

try:
    cn_cand_pool = []
    ne_cand_pool = []
    p_cand_pool = []
    v_cand_pool = []
    if args.get_set == 'yes':
        print 'loading cand pool...'
    	cn_cand_pool = read_list('./cand_pool/cn_cand_pool.txt')
    	ne_cand_pool = read_list('./cand_pool/ne_cand_pool.txt')
    	p_cand_pool = read_list('./cand_pool/p_cand_pool.txt')
    	v_cand_pool = read_list('./cand_pool/v_cand_pool.txt')
        print 'load complete'
    else:
        print 'making cand pool...'
    	# Reading Data, dataset.py
    	if args.small: # if given --small in arg, then limit 1000 set
           train_cn_exs = read_cbt(train_cn_path, limit=1000)
           train_ne_exs = read_cbt(train_ne_path, limit=1000)
           train_p_exs = read_cbt(train_p_path, limit=1000)
           train_v_exs = read_cbt(train_v_path, limit=1000)
    	else:
            train_cn_exs = read_cbt(train_cn_path)
            train_cn_exs.extend(read_cbt(test_cn_path))
            print 'done1'
            train_ne_exs = read_cbt(train_ne_path)
            train_ne_exs.extend(read_cbt(test_ne_path))
            print 'done2'
            train_p_exs = read_cbt(train_p_path)
            train_p_exs.extend(read_cbt(test_p_path))
            print 'done3'
            train_v_exs = read_cbt(train_v_path)
            train_v_exs.extend(read_cbt(test_v_path))
            print 'done4'

            # build candidate pool
            for cn_ex in train_cn_exs:
                for ex_cand in cn_ex['candidate']:
                    ex_cand = ex_cand.lower()
                    if ex_cand not in cn_cand_pool:
                       cn_cand_pool.append(ex_cand)
            write_list(cn_cand_pool , 'cn_cand_pool')
            print 'cn cand pool done'

            for ne_ex in train_ne_exs:
                for ex_cand in ne_ex['candidate']:
                    ex_cand = ex_cand.lower()
                    if ex_cand not in ne_cand_pool:
                        ne_cand_pool.append(ex_cand)
            write_list(ne_cand_pool , 'ne_cand_pool')
            print 'ne cand pool done'

            for p_ex in train_p_exs:
                for ex_cand in p_ex['candidate']:
                    ex_cand = ex_cand.lower()
                    if ex_cand not in p_cand_pool:
                        p_cand_pool.append(ex_cand)
            write_list(p_cand_pool , 'p_cand_pool')
            print 'p cand pool done'

            for v_ex in train_v_exs:
                for ex_cand in v_ex['candidate']:
                    ex_cand = ex_cand.lower()
                    if ex_cand not in v_cand_pool:
                        v_cand_pool.append(ex_cand)
            write_list(v_cand_pool , 'v_cand_pool')
            print 'v cand pool done'

    test_exs = read_test_set(test_path)
    # sperate test_set
    test_cn_exs = []
    test_ne_exs = []
    test_p_exs  = []
    test_v_exs  = []
    cn_id = []
    ne_id = []
    p_id = []
    v_id = []
    count = 1 #record id of ex

    # for each story ex
    for ex in test_exs:
        # determine ex is belong to which kind of question
        for ex_cand in ex['candidate']:
            ex_cand = ex_cand.lower()
            if ex_cand in cn_cand_pool:
                test_cn_exs.append(ex)
                cn_id.append(count)
                break
            elif ex_cand in ne_cand_pool:
                test_ne_exs.append(ex)
                ne_id.append(count)
                break
            elif ex_cand in p_cand_pool:
                test_p_exs.append(ex)
                p_id.append(count)
                break
            elif ex_cand in v_cand_pool:
                test_v_exs.append(ex)
                v_id.append(count)
                break
        else: # finish for loop but no match
            test_ne_exs.append(ex)
            ne_id.append(count)
        count += 1

    param_b = args.window_b    
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

# loading 4 models
print 'loading cn model from ' + path.join('model', args.model, 'cn_learner')
with open(path.join('model', args.model, 'cn_learner'), 'rb') as dill_file:
    cn_learner = dill.load(dill_file)
print 'loading ne model from ' + path.join('model', args.model, 'ne_learner')
with open(path.join('model', args.model, 'ne_learner'), 'rb') as dill_file:
    ne_learner = dill.load(dill_file)
print 'loading p model from ' + path.join('model', args.model, 'p_learner')
with open(path.join('model', args.model, 'p_learner'), 'rb') as dill_file:
    p_learner = dill.load(dill_file)
print 'loading cn model from ' + path.join('model', args.model, 'v_learner')
with open(path.join('model', args.model, 'v_learner'), 'rb') as dill_file:
    v_learner = dill.load(dill_file)
print 'load models complete'

# start test
print 'testing cn model...'
cn_ans = pred(cn_learner,test_cn_exs)
print 'testing ne model...'
ne_ans = pred(ne_learner,test_ne_exs)
print 'testing p model...'
p_ans = pred(p_learner,test_p_exs)
print 'testing cn model...'
v_ans = pred(v_learner,test_v_exs)

# connect 4 ans_lists and id_lists 
output_dict = dict(zip(cn_id,cn_ans))
output_dict.update(zip(ne_id,ne_ans))
output_dict.update(zip(p_id,p_ans))
output_dict.update(zip(v_id,v_ans))

# sort by id
output_orderDict = collections.OrderedDict(sorted(output_dict.items()))
for k, v in output_orderDict.iteritems(): print k,v
write_csv('./final_output/answer.csv', output_orderDict)
