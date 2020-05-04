import pickle
import string
from nltk.corpus import stopwords

def msg_processor(mess):
    no_punc = [c for c in mess if c not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

def Prediction(mess):
    mess = [mess]
    pickle_f = open('pipeline.pickle','rb')
    trans = pickle.load(pickle_f)
    pred = trans.predict(mess)
    pred = ''.join(pred)
    if pred == 'ham':
        print("Normal Message")
    else:
        print("It's a Spam Message")

Prediction(input())
