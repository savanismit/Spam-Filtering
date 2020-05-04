import nltk
import pandas as pd
import pickle
import string 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TfT
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as Pp
from sklearn.metrics import classification_report

def msg_processor(mess):
    no_punc = [c for c in mess if c not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
        
#Reading CSV File
msgs = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])

#Using PipeLine Method
pp = Pp([('X_train',CountVectorizer(analyzer=msg_processor)),('mess_trans',TfT()),('classfier',MultinomialNB()),])

pre_fit = pp.fit(msgs['message'],msgs['label'])

with open("pipeline.pickle","wb") as f:
    pickle.dump(pre_fit,f)

