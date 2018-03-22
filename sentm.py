from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import random
import os
import unicodedata
from TwitterSearch import *


COUNT_VECT = "{base_path}/sklearn-models/count_vect.pkl".format(
    base_path=os.path.abspath(os.path.dirname(__file__))
)

TF_TRANS = "{base_path}/sklearn-models/tf_transformer.pkl".format(
    base_path=os.path.abspath(os.path.dirname(__file__))
)

CLF = "{base_path}/sklearn-models/clf.pkl".format(
    base_path=os.path.abspath(os.path.dirname(__file__))
)

raw_data = list()
data = list()
label = list()

with open('training.txt') as f:
    for line in f.readlines():
        line = [int(line[:1]), line[1:].strip()]
        raw_data.append(line)
with open('amazon_cells_labelled.txt') as f:
    for line in f.readlines():
        line = line.strip()
        line = [int(line[-1:]), line[:-1].strip()]
        raw_data.append(line)
with open('imdb_labelled.txt') as f:
    for line in f.readlines():
        line = line.strip()
        line = [int(line[-1:]), line[:-1].strip()]
        raw_data.append(line)
with open('yelp_labelled.txt') as f:
    for line in f.readlines():
        line = line.strip()
        line = [int(line[-1:]), line[:-1].strip()]
        raw_data.append(line)

random.shuffle(raw_data)

for item in raw_data:
    label.append(item[0])
    data.append(item[1])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)

tf_transformer = TfidfTransformer(use_idf=True)
X_train_tf = tf_transformer.fit_transform(X_train_counts)

clf = RidgeClassifier().fit(X_train_tf[:9000], label[:9000])

joblib.dump(count_vect, 'count_vect.pkl')
joblib.dump(tf_transformer, 'tf_transformer.pkl')
joblib.dump(clf, 'clf.pkl')

predicted = clf.predict(X_train_tf[9000:])

def analysis_sentence(lines):
    print (lines)
    count_vect = joblib.load(COUNT_VECT) 
    tf_transformer = joblib.load(TF_TRANS) 
    clf = joblib.load(CLF)

    x_test_counts = count_vect.transform([lines])
    x_test_tf = tf_transformer.transform(x_test_counts)

    predicted = clf.predict(x_test_tf)
    #print ('predicted', predicted)
    return predicted[0]

def hello():
    goodt=[]
    badt=[]
    check=2
    try:
            tso = TwitterSearchOrder()
            tso.set_keywords(['@railminindia'])
    #	    tso.set_language('de')
            tso.set_include_entities(False)
            ts = TwitterSearch(
                    consumer_key = 'BqJsOM1X97mvQf8P9ZQWnHtfL',
                    consumer_secret = '6O8l2O5kg1urEoyWuMxFoDL5FeGBBp1q4fwuoNd6SqCpd798hk',
                    access_token = '3266138964-vR5neysWkZYYDiWPJF8Ryh7KQxEYDak953QWgB5',
                    access_token_secret = '5WWvSXIDgx3H78vmgSuWxSXgdAVuAKiLJMvjeAYtj5DVG'
            )
            count=0

            for tweet in ts.search_tweets_iterable(tso):
                count=count+1;
                if count<=10:
                    # print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
                    check=analysis_sentence(tweet['text'])
                    if check==1:
                        goodt.append(str(tweet['text']))
                    if check==0:
                        badt.append(str(tweet['text']))
            print len(goodt)
            print "----------------"
            print len(badt)
                else:
                    break
    except TwitterSearchException as e:
        print(e)

    return ("Done")

print(accuracy_score(label[9000:]))#, predicted))
