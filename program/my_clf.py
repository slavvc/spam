from sklearn.base import BaseEstimator
import numpy as np

class Classifier(BaseEstimator):
    def __init__(self, return_cat=False):
        self.return_cat = return_cat
        self.model = None
        
    def count_words(self, vector):
        counter = {}
        for i, w in zip(vector.indices, vector.data):
            if i in counter:
                counter[i] += w
            else:
                counter[i]  = w
        return counter

    def add_counters(self, c1, c2):
        res = {}
        for k, v in c1.items():
            res[k] = v
        for k, v in c2.items():
            if k in res:
                res[k] += v
            else:
                res[k]  = v
        return res

    def add_counters_inplace(self, c1, c2):
        for k, v in c2.items():
            if k in c1:
                c1[k] += v
            else:
                c1[k] = v
    def total_counts(self, lines):
        counters = {
                'ham' : {},
                'spam': {}
                }
        sizes = {
                'ham' : 0.,
                'spam': 0.
                }
        for kind, line in lines:
            #counters[kind] = add_counters(counters[kind], count_words(line))
            self.add_counters_inplace(counters[kind], self.count_words(line))
            sizes[kind] += 1

#         counters = {kind: self.filter_counter(counter) for kind, counter in counters.items()}

        for kind, counter in counters.items():
            size = sizes[kind]
            for word in counter:
                counter[word] /= size
        return counters, sizes

    def fit(self, X, y):
        self.model = self.total_counts(zip(y, X))
    
    def guess(self, line, model):
        counters, sizes = model
        p_ham  = sizes['ham']
        p_spam = sizes['spam']
        eps = 1e-5
        for i,w in zip(line.indices, line.data):
            p_in_ham  = counters['ham' ][i]**w if i in counters['ham' ] else eps
            p_in_spam = counters['spam'][i]**w if i in counters['spam'] else eps
            p_sum = p_in_ham + p_in_spam
            p_ham  *= p_in_ham / p_sum
            p_spam *= p_in_spam / p_sum
    #    if p_ham+p_spam == 0:
    #        print(line)
        return p_ham / (p_ham + p_spam) if p_ham+p_spam > 0.0 else 0.5
    
    def predict(self, X):
        if self.model is None:
            raise 'wasn\'t fit'
        if self.return_cat:
            r = ['ham' if self.guess(x, self.model) > 0.5 else 'spam' for x in X]
        else:
            r = [self.guess(x, self.model) for x in X]
        return np.array(r)
    def decision_function(self, X):
        if self.model is None:
            raise 'wasn\'t fit'
        r = np.array([self.guess(x, self.model) for x in X])
        return r
    