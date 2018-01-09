# basic embeding models as baselines.
from cbtest.common import *
from cbtest.layers import (log, dot, mean, softmax, Embed, floatX, stack, LSTM, MemoryLayer, LinearLayer, position_encoding, LSTMq)
from cbtest.utils import choice, Timer
from cbtest.evaluate import (accuracy, disagree)
from cbtest.dataset import remove_stopwords, lower, remove_punctuation, filter, unkify
import cbtest.optimizers as optimizers

import theano.tensor as T


class CBTLearner(object):
    def __init__(self, batchsize=1, hidden_dim=100, lr=1e-4, sen_maxlen=128, flags={}, **kwargs):
        '''
        a sentence is a list of lexicals.
        a context is a list of sentences.
        '''
        self.num_candidate = 10
        self.sen_maxlen = 128
        self.mem_size = 1024
        self.unit_size = 128

        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.flags = {}
        self.kwargs = kwargs


    def preprocess_sentence(self, sentence):
        '''
        preprocess training sentences.
        '''
        # return lower(remove_stopwords(sentence))
        return lower(sentence)


    def preprocess_dataset(self, exs):
        def preprocess(sentence):
            sen = unkify(self.preprocess_sentence(sentence), self.vocab)
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
            ex['answer'] = preprocess([ex['answer']])[0] # make sure answer is a word(string)


    def create_vocab(self, exs):

        # word pool for all word in training data
        vocab = {
            '<null>': 0, # no word in current position.
            '<unk>': 1,  # unknown word.
            'xxxxx': 2,  # blank in query.
        }

        def add_word_if_not_exist(word):
            if word not in vocab:
                vocab[word] = len(vocab) # give each word a num in order to create ivocab?

        def add_sen_if_not_exist(sentence):
            sentence = self.preprocess_sentence(sentence) #return lower case
            for word in sentence: # add word into vocab
                add_word_if_not_exist(word)
            if len(sentence) > self.sen_maxlen: # update max len of all sentences 
                self.sen_maxlen = len(sentence)

        # Take out each story from list, each story is a dictionary
        # Each story contains context(20 setences, a list of 20 list), query(1 sentence, list), cand(10 words, list), answer(1 word, string)
        # Note that each sentence is a list of words
        # Adding words(not repeat) into vocab
        for ex in exs:
            for sen in ex['context']:
                add_sen_if_not_exist(sen)
            add_sen_if_not_exist(ex['query'])
            add_sen_if_not_exist(ex['candidate'])

        vocab_size = len(vocab)
        print '[vocab size]', vocab_size
        print '[max sentence length]', self.sen_maxlen
        print '[num train exs]', len(exs)
        print '[batchsize]', self.batchsize

        self.vocab = vocab
        self.ivocab = {val: key for (key ,val) in self.vocab.items()} 
        self.vocab_size = vocab_size


    def encode_context(self, ex):
        raise NotImplementedError()


    def encode_query(self, ex):
        raise NotImplementedError()


    def encode_label(self, ex):
        return self.vocab[ex['answer']]


    def encode_candidate(self, ex):
        cv = [self.vocab[word] for word in ex['candidate']]
        return cv


    def arch(self):
        raise NotImplementedError()


    def encode_minibatch(self, minibatch):
        contexts = []
        querys = []
        yvs = []
        cvs = []
        for ex in minibatch:
            query = self.encode_query(ex)
            contexts.append(self.encode_context(ex)) # return a array(n*n)
            querys.append(self.encode_query(ex)) # return an array(1 row)
            yvs.append(self.encode_label(ex)) # return a number , index of answer in vocab
            cvs.append(self.encode_candidate(ex)) # return a list of number, each is index of each candidate in vocab
        contexts = np.vstack([context[np.newaxis, ...] for context in contexts])
        querys = np.array(querys, dtype=np.int64)
        yvs = np.array(yvs, dtype=np.int64)
        cvs = np.array(cvs, dtype=np.int64)

        return (contexts, querys, cvs, yvs)


    def arch_memnet_lexical(self):
        '''
        each memory slot is a lexical.
        '''
        contexts = T.ltensor3('contexts')
        querys = T.lmatrix('querys')
        yvs = T.lvector('yvs')
        hop = 1

        params = []
        question_layer = Embed(self.vocab_size, self.hidden_dim) # layer.py

        # q is a tensor reshape from question_layer.W , batchsize * sen_maxlen * hidden_dim 
        q = T.reshape(question_layer(querys.flatten()),
                      (self.batchsize, self.sen_maxlen, self.hidden_dim)
                      )
        if self.kwargs.get('position_encoding'):
            lmat = position_encoding(self.sen_maxlen, self.hidden_dim).dimshuffle('x', 0, 1)
            print '[memory network] use PE'
            q = q * lmat

        # u is a vector of means(64 means of each sentence) from 64(batchsize) stories
        u = mean(q, axis=1)
        params.extend(question_layer.params)

        mem_layers = []
        for hi in range(hop):
            mem_layer = MemoryLayer(self.batchsize, self.mem_size, self.unit_size, self.vocab_size, self.hidden_dim,
                                    **self.kwargs)# define mean
            params.extend(mem_layer.params)
            mem_layers.append(mem_layer)

            # o is weighted sum 
            o = mem_layer(contexts, u)
            u = u + o

        linear = LinearLayer(self.hidden_dim, self.vocab_size)
        params.extend(linear.params)

        # Take the output from memory and do inner product through weight matrix
        probs = softmax(linear(u))
        inputs = {
            'contexts': contexts,
            'querys': querys,
            'yvs': yvs,
            'cvs': T.lmatrix('cvs')
        }
        return (probs, inputs, params)


    def encode_context_lexical(self, ex):
        '''
        lexical encoding method. each memory slot is a word.
        '''
        enc = np.zeros((self.mem_size, self.unit_size), dtype=np.int64)
        context = ex['context']
        ei = 0
        for sentence in context:
            for word in sentence:
                enc[ei, 0] = self.vocab[word]
                ei += 1
                if ei >= self.mem_size:
                    print '[warning] network out of memory'
                    return enc[:self.mem_size]
        return enc


    def encode_query_lexical(self, ex):
        '''
        lexical encoding method.
        '''
        enc = np.zeros(self.sen_maxlen, dtype=np.int64)
        ei = 0
        for word in ex['query']:
            enc[ei] = self.vocab[word]
            ei += 1
        return enc


    def encode_query_window(self, ex, param_b=2):
        '''
        a window of size 2 * b + 1 centered around the missing word.
        '''
        ind = ex['query'].index('xxxxx') + param_b
        query = [0] * param_b + [self.vocab[w] for w in ex['query']] + [0] * param_b
        window = query[ind - param_b : ind + param_b + 1]
        return np.array(window, dtype=np.int64)


    def encode_context_window(self, ex, param_b=2):
        '''
        search in context occurrances of candidate words.
        and extract windows of size 2 * b + 1 around these words.
        '''
        candidates = set(ex['candidate'])
        context = sum(ex['context'], []) # every slot in context is a list currently. we need to make the whole context become a list, each block is word
        inds = []
        for (ind, word) in enumerate(context):
            if word in candidates:
                inds.append(ind + param_b) # param_b are added as  offset for padding 0
        context = [0] * param_b + [self.vocab[w] for w in context] + [0] * param_b  # vocab return the index of that word in vocab
        res = np.zeros((self.mem_size, self.unit_size), dtype=np.int64)
        for (ei, ind) in enumerate(inds):
            res[ei, :] = context[ind - param_b : ind + param_b + 1]
            ei += 1
            if ei >= self.mem_size:
                print '[warning] network out of memory'
                break
        return np.array(res, dtype=np.int64)


    def arch_lstmq(self, param_b=2):

        contexts = T.ltensor3('contexts')
        querys = T.lmatrix('querys')
        yvs = T.lvector('yvs')

        params = []
        question_layer = Embed(self.vocab_size, self.hidden_dim)
        params.extend(question_layer.params)
        q = T.reshape(question_layer(querys.flatten()),
                      (self.batchsize, self.sen_maxlen, self.hidden_dim)
                      )
        lmat = position_encoding(self.sen_maxlen, self.hidden_dim).dimshuffle('x', 0, 1)
        q = q * lmat
        u = mean(q, axis=1)


        embed_layer = Embed(self.vocab_size, self.hidden_dim)
        params.extend(embed_layer.params)
        lmat = position_encoding(self.unit_size, self.hidden_dim).dimshuffle('x', 'x', 0, 1)
        m = T.reshape(embed_layer(contexts.flatten()), (self.batchsize, self.mem_size, self.unit_size, self.hidden_dim))
        m = mean(m * lmat, axis=2)

        lstm = LSTMq(self.batchsize, self.hidden_dim)
        params.extend(lstm.params)
        o = lstm(m.dimshuffle(1, 0, 2), u)

        linear = LinearLayer(self.hidden_dim, self.vocab_size)
        params.extend(linear.params)
        probs = softmax(linear(o))

        inputs = {
            'contexts': contexts,
            'querys': querys,
            'yvs': yvs,
            'cvs': T.lmatrix('cvs')
        }
        return (probs, inputs, params)


    def encode_query_sentence(self, ex):
        enc = np.zeros(self.sen_maxlen, dtype=np.int64)
        ei = 0
        for word in ex['query']:
            enc[ei] = self.vocab[word]
            ei += 1
            if ei >= self.sen_maxlen:
                print '[warning] query out of memory'
                break
        return enc


    def encode_context_sentence(self, ex):
        enc = np.zeros((self.mem_size, self.unit_size), dtype=np.int64)
        for (si, sentence) in enumerate(ex['context']):
            if si >= self.mem_size:
                print '[warning] network out of memory si'
                break
            ei = 0
            for word in sentence:
                enc[si, ei] = self.vocab[word]
                ei += 1
                if ei >= self.sen_maxlen:
                    print '[warning] network out of memory ei'
                    break
        return enc


    def encode_query_selfsup(self, ex, param_b=2):
        return self.encode_query_window(ex, param_b=param_b)


    def encode_context_selfsup(self, ex, param_b=2):
        return self.encode_context_window(ex, param_b=param_b)


    def encode_label_selfsup(self, ex, param_b=2):
        contexts = self.encode_context_selfsup(ex, param_b=param_b)
        answer = self.vocab[ex['answer']]
        targets = [int(context[param_b] == answer) for context in contexts]
        return np.array(targets, dtype=np.int64)


    def encode_candidate_selfsup(self, ex, param_b=2):
        contexts = self.encode_context_selfsup(ex, param_b=param_b)
        return np.array([context[param_b] for context in contexts], dtype=np.int64)

    def arch_memnet_selfsup(self):
        '''
        memory net with self supervision.
        '''
        contexts = T.ltensor3('contexts')
        querys = T.lmatrix('querys')
        yvs = T.lmatrix('yvs')

        params = []
        question_layer = Embed(self.vocab_size, self.hidden_dim)
        q = T.reshape(question_layer(querys.flatten()),
                      (self.batchsize, self.sen_maxlen, self.hidden_dim)
                      )
        if self.kwargs.get('position_encoding'):
            lmat = position_encoding(self.sen_maxlen, self.hidden_dim).dimshuffle('x', 0, 1)
            print '[memory network] use PE'
            q = q * lmat
        u = mean(q, axis=1)
        params.extend(question_layer.params)

        mem_layer = MemoryLayer(self.batchsize, self.mem_size, self.unit_size, self.vocab_size, self.hidden_dim,
                                **self.kwargs)
        probs = mem_layer.get_probs(contexts, u).dimshuffle(0, 2)

        inputs = {
            'contexts': contexts,
            'querys': querys,
            'yvs': yvs,
            'cvs': T.lmatrix('cvs')
        }

        # probs -> predicted answer, inputs -> theano var, params -> a shitload of rand vector
        return (probs, inputs, params)

    def loss_selfsup(self, probs, yvs):
        return -mean(log(T.sum(probs * yvs, axis=1)))


    def loss(self, probs, yvs):
        return -mean(log(probs[T.arange(self.batchsize), yvs]))


    def compile(self):
        (probs, inputs, params) = self.arch()

        contexts = inputs['contexts']
        querys = inputs['querys']
        yvs = inputs['yvs']
        cvs = inputs['cvs']

        # build forward propagation.
        self.fprop = theano.function(inputs=[contexts, querys, cvs], outputs=probs,
                                     on_unused_input='ignore')

        loss = self.loss(probs, yvs)

        # build backward propagation.
        updates = optimizers.Adam(loss, params, alpha=FX(self.lr))
        self.bprop = theano.function(inputs=[contexts, querys, cvs, yvs],
                                        outputs=loss, updates=updates,
                                     on_unused_input='ignore')


    def sample_minibatch(self, exs):
        minibatch = choice(exs, size=self.batchsize, replace=True)
        meta = {}
        meta['minibatch'] = minibatch
        (contexts, querys, cvs, yvs) = self.encode_minibatch(minibatch)

        return (contexts, querys, cvs, yvs, meta)


    def train(self, exs, num_iter=100):
        self.preprocess_dataset(exs)

        for it in range(num_iter):
            for ni in range(len(exs) / self.batchsize + 1):
                (contexts, querys, cvs, yvs, meta) = self.sample_minibatch(exs)

                error = self.bprop(contexts, querys, cvs, yvs)
                if ni % 10 == 0:
                    print 'iter', ni * self.batchsize, '/', len(exs), 'error', error


    def test_selfsup(self, exs):
        self.preprocess_dataset(exs)

        all_preds = []
        all_truths = []
        for offset in range(0, len(exs), self.batchsize):
            minibatch = exs[offset:offset + self.batchsize]
            while len(minibatch) < self.batchsize:
                minibatch.append(minibatch[-1])
            (contexts, querys, cvs, _) = self.encode_minibatch(minibatch)
            yvs = np.array([self.vocab[ex['answer']] for ex in minibatch], dtype=floatX)
            preds = []
            probs = self.fprop(contexts, querys, cvs)
            for bi in range(self.batchsize):
                candidates = set(cvs[bi, :])
                if 0 in candidates:
                    candidates.remove(0)
                prob_by_candidate = {c: sum(probs[bi, cvs[bi, :] == c]) for c in candidates}
                preds.append(max(prob_by_candidate, key=lambda c: prob_by_candidate[c]))
            truths = np.array(yvs, dtype=np.int64)
            if sum(truths == 1):
                print '[warning] unknown word in answer'
            truths[truths == 1] = -1
            all_preds.extend(preds[:min(self.batchsize, len(exs)-offset)])
            all_truths.extend(truths[:min(self.batchsize, len(exs)-offset)])
        acc = accuracy(all_preds, all_truths)
        errs = disagree(all_preds, all_truths)
        return (acc, errs)


    def test(self, exs):
        self.preprocess_dataset(exs)

        all_preds = []
        all_truths = []
        for offset in range(0, len(exs), self.batchsize):
            minibatch = exs[offset:offset + self.batchsize]
            while len(minibatch) < self.batchsize:
                minibatch.append(minibatch[-1])
            (contexts, querys, cvs, yvs) = self.encode_minibatch(minibatch)
            yvs = np.array(yvs).astype(theano.config.floatX)
            inds = np.argmax(self.fprop(contexts, querys, cvs)
                              [np.transpose([range(cvs.shape[0])] * cvs.shape[1]),
                               cvs], axis=1)
            preds = cvs[range(self.batchsize), inds]
            truths = np.array(yvs, dtype=np.int64)
            if sum(truths == 1):
                print '[warning] unknown word in answer'
            truths[truths == 1] = -1
            all_preds.extend(preds[:min(self.batchsize, len(exs)-offset)])
            all_truths.extend(truths[:min(self.batchsize, len(exs)-offset)])
        acc = accuracy(all_preds, all_truths)
        errs = disagree(all_preds, all_truths)
        return (acc, errs)


