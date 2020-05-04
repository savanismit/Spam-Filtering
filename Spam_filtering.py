import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TfT
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def msg_processor(mess):
    no_punc = [c for c in mess if c not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    
msgs = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])
msgs['len'].plot(bins=150,kind='hist')
msgs.hist(column='len',by='label',bins=150,figsize=(10,4))
Tfidf_trans = TfT().fit(mess_trans)

X_train, X_test, y_train, y_test = train_test_split(msgs['message'],msgs['label'], test_size=0.2, random_state=20)

transformer = CountVectorizer(analyzer=msg_processor).fit(X_train)
mess_trans = tranformer.transform(X_train)
Tfid_trans = TfT().fit(mess_trans)

mess_idf = Tfid_trans.transform(mess_trans)
spam_model = MultinomialNB().fit(mess_idf,y_train)

transformer_test = CountVectorizer(analyzer=msg_processor).fit(X_test)
mess_trans_test = tranformer.transform(X_test)

all_pred_test = spam_model.predict(mess_trans_test)
print(classification_report(y_test,all_pred_test))
