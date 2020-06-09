
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, help='номер комбинации предобработок')
    parser.add_argument(
        '-classifier', choices=['bayes', 'logistic', 'my'], 
        required=True, help='модель классификатора [bayes, logistic, my]', 
        metavar='классификатор'
    )
    parser.add_argument('-dataset', type=str, required=True, help='файл датасета', metavar='датасет')
    parser.add_argument('-output', type=str, required=True, help='результат обучения', metavar='модель')
    
    args = parser.parse_args()
    

    import nltk
    import html

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression

    from sklearn.pipeline import make_pipeline

    import numpy as np
    import pickle
    
    from get_words import Get_words
    from my_clf import Classifier


    with open(args.dataset, 'rt', encoding='utf-8') as f:
        lines = [[html.unescape(y.strip()) for y in x.split('\t')] for x in f.readlines()]

    st, bigram, stem, lemm, alpha, tfidf = (x=='1' for x in '{:6b}'.format(args.n))


    if args.classifier == 'bayes':
        classifier = MultinomialNB()
    elif args.classifier == 'logistic':
        classifier = LogisticRegression()
    elif args.classifier == 'my':
        classifier = Classifier(return_cat=True)
    filter_f = Get_words(st, bigram, stem, lemm, alpha).func
    vectorizer = TfidfVectorizer(analyzer=filter_f)\
        if tfidf\
        else CountVectorizer(analyzer=filter_f)
        
        
    X = [
        l[1]
        for l in lines
    ]
    y = [l[0] for l in lines]

    pipe = make_pipeline(vectorizer, classifier)

    pipe.fit(X,y)

    with open(args.output, 'wb') as f:
        pickle.dump(pipe, f)