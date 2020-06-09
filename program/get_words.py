import nltk
import spacy
#nltk.download('punkt')
import re

sp = spacy.load('en', disable=['parser', 'ner'])

class Get_words(object):
    def __init__(
        self, 
        stopwords=False, 
        bigrams=False, 
        stemming=False, 
        lemming=False, 
        alpha=False
    ):
        self.stopwords = stopwords
        self.bigrams = bigrams
        self.stemming = stemming
        self.lemming = lemming
        self.alpha = alpha
    def func(self, line):
        if self.lemming:
            words = [token.lemma_ for token in Get_words.lemmer(line)]
        else:
            words = Get_words.re_words.findall(line)#nltk.tokenize.word_tokenize(line)
        words = [
            w.lower()
            for w in words 
            if (any(c.isalpha() for c in w) if self.alpha else True)
            and (w.lower() not in Get_words.stopwords if self.stopwords else True)
        ]
        if self.stemming:
            words = [Get_words.stemmer.stem(word) for word in words]  
        if self.bigrams:
            words = list(nltk.bigrams(words))
        return words
Get_words.lemmer = spacy.load('en', disable=['parser', 'ner'])
Get_words.stemmer = nltk.stem.snowball.SnowballStemmer('english')
Get_words.stopwords = set(nltk.corpus.stopwords.words('english'))
Get_words.re_words = re.compile(
    r'(?:[a-zA-Z]+)|(?:\d|\.|,)*(?:\d)+|(?=\W+)(?=\S+).+?(?=\s|\w|$)'
)
