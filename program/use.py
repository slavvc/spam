#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, type=str, help='файл сохраненной модели', metavar='модель')
    parser.add_argument('-message', required=True, type=str, help='сообщение для классификации', metavar='сообщение')

    args = parser.parse_args()
 
    import pickle
    from get_words import Get_words
    from my_clf import Classifier

    with open(args.model, 'rb') as f:
        clf = pickle.load(f)
    y = clf.predict([args.message])[0]
    print(y)
    
