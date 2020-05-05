from flask import Flask,request
import pickle
import string
from nltk import corpus 

app=Flask(__name__)
@app.route('/')
def hello():
    return 'Hello Smit'

def msg_processor(mess):
    no_punc = [c for c in mess if c not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in corpus.stopwords.words('english')]

@app.route('/<string:msg>',methods=['GET','POST'])
def test(msg):
    mess = [msg]
    pickle_f = open('pipeline.pickle','rb')
    trans = pickle.load(pickle_f)
    pred = trans.predict(mess)
    pred = ''.join(pred)
    if pred == 'ham':
        return "Normal Message"
    else:
        return "It's a Spam Message"

if __name__ == '__main__':
    app.run(debug=True)