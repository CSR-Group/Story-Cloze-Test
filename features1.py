import csv 
from sklearn.model_selection import train_test_split
from nltk.util import ngrams
import vsmlib
import torch
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import random
import numpy as np
from vaderSentiment import vaderSentiment
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy
import matplotlib.pyplot as plt

path_to_vsm = "word_linear_glove_500d"
vsm = vsmlib.model.load_from_dir(path_to_vsm)

def getEncoding(word): 
    if(vsm.has_word(word)):
        return vsm.get_row(word)
    else:
        return np.zeros(500)  

def diffAvgWE(str1):
    str1 = [getEncoding(word) for word in word_tokenize(str1.lower())] 
    #str2 = [getEncoding(word) for word in word_tokenize(str2.lower())]
    avg1 = sum(str1) / len(str1)
    #avg2 = sum(str2) / len(str2)
    #print(avg1)
    return avg1
    #return numpy.linalg.norm(avg1 - avg2)

def ngrams(str1, str2):
    print(str1)
    s = str1.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, 5))
    print(output)
    
def cosineSim (X,Y):
    
    # tokenization 
    X_list = word_tokenize(X.lower())  
    Y_list = word_tokenize(Y.lower()) 
    
    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[] 
    
    # remove stop words from string 
    X_set = {w for w in X_list if not w in sw}  
    Y_set = {w for w in Y_list if not w in sw} 
    
    # form a set containing keywords of both strings  
    rvector = X_set.union(Y_set)  
    for w in rvector: 
        if w in X_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in Y_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return (cosine) 

def getBigramCount(str1, str2):
    bigrm1 = list(nltk.bigrams(str1.split()))
    bigrm2 = list(nltk.bigrams(str2.split()))

    count = 0
    for bigram in bigrm1:
        if bigram in bigrm2:
            count+=1
    return count

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    sim = float(len(c)) / (len(a) + len(b) - len(c))
    #print(sim)
    return sim

def readData(filename, rnn_filename):
    data = []
    input = []
    output = []
    ids = []
    #sentiment  = []
    with open(filename, 'r', encoding="utf-8") as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvdata)
        analyzer = vaderSentiment.SentimentIntensityAnalyzer()
        for line in csvdata:
            d = []
            i = []
            ids.append(line[0])

            # Sentiment Features
            # i.append(line[1])
            # sentiment = analyzer.polarity_scores(line[1])
            # d.append(sentiment['neg'])
            # d.append(sentiment['neu'])
            # d.append(sentiment['pos'])

            # i.append(line[2])
            # sentiment = analyzer.polarity_scores(line[2])
            # d.append(sentiment['neg'])
            # d.append(sentiment['neu'])
            # d.append(sentiment['pos'])
            
            # i.append(line[3])
            # sentiment = analyzer.polarity_scores(line[3])
            # d.append(sentiment['neg'])
            # d.append(sentiment['neu'])
            # d.append(sentiment['pos'])

            # i.append(line[4])
            # sentiment = analyzer.polarity_scores(line[4])
            # d.append(sentiment['neg'])
            # d.append(sentiment['neu'])
            # d.append(sentiment['pos'])

            i.append(line[5])
            sentiment = analyzer.polarity_scores(line[5])
            d.append(sentiment['neg'])
            d.append(sentiment['neu'])
            d.append(sentiment['pos'])

            i.append(line[6])
            sentiment = analyzer.polarity_scores(line[6])
            d.append(sentiment['neg'])
            d.append(sentiment['neu'])
            d.append(sentiment['pos'])

            #avg WE
            # d.append(diffAvgWE(line[1] + " " + line[2] + " " + line[3] + " " + line[4],line[5]))
            # d.append(diffAvgWE(line[1] + " " + line[2] + " " + line[3] + " " + line[4],line[6]))
            e1 = diffAvgWE(line[5])
            for e in e1:
                d.append(e)
            e2 = diffAvgWE(line[6])
            for e in e2:
                d.append(e)

            # Ngram features
            #d.append(getBigramCount(line[5],line[6]))

            #Length of endings
            e1 = line[5].split(' ')
            d.append(len(e1))
            e2 = line[6].split(' ')
            d.append(len(e2))

            #Cosine Similarity
            d.append(cosineSim(line[1] + line[2] + line[3] + line[4],line[5]))
            d.append(cosineSim(line[1] + line[2] + line[3] + line[4],line[6]))

            d.append(get_jaccard_sim(line[1] + line[2] + line[3] + line[4],line[5]))
            d.append(get_jaccard_sim(line[1] + line[2] + line[3] + line[4],line[6]))

            data.append(d)
            input.append(i)
            if(filename != "test.csv"):
                output.append(line[7])

    # add RNN features
    with open(rnn_filename, 'r', encoding="utf-8") as f:
        csvdata = csv.reader(f, delimiter=',', quotechar='"')
        count = 0
        index = 0
        for line in csvdata:
            if count%2==0:
                if rnn_filename == "predtest.csv":
                    data[index].append(float(line[0]))
                else:
                    data[index].append(float(line[2]))
            else:
                if rnn_filename == "predtest.csv":
                    data[index].append(float(line[0]))
                else:
                    data[index].append(float(line[2]))
                index += 1
            count += 1
    
    print(data[10])
    return data, output, ids

def getTrainingAndValData(data, output, size):
    X_train, X_test, Y_train, Y_test = train_test_split(data, output, test_size = size, random_state=1234)
    return X_train, X_test, Y_train, Y_test

def preprocessData(data, label, isTrain):
    d = []
    l = []

    if isTrain:
        data_1 = []
        data_0 = []
        for i in range(len(data)):
            if label[i]=='1':
                data_1.append(data[i])
            else:
                data_0.append(data[i])
        
        random.shuffle(data_0)
        data_0 = data_0[:len(data_1)]
    
        data = []
        data = data_0
        data = data + data_1
        label = [1 for i in range(len(data_1)*2)]
        label[:len(data_1)] = [0] * len(data_1)

    for i in range(len(data)):
        seq = []
        tokens = data[i][0].split(' ')

        for word in tokens:
            word = re.sub('[^A-Za-z0-9]+', '', word).lower()
            seq.append(getEncoding(word))
        d.append(torch.from_numpy(np.array([seq])).type(torch.FloatTensor))
        l.append(torch.from_numpy(np.array(int(label[i]))).type(torch.FloatTensor))

    return d, l

def main():
    train_data, train_labels, train_ids = readData("train.csv","train4.csv")
    dev_data, dev_labels, dev_ids = readData("dev.csv","dev4.csv")
    test_data, test_labels, test_ids = readData("test.csv","predtest.csv")

    svclassifier = SVC(kernel='poly', degree=3)
    svclassifier.fit(train_data, train_labels)
    dev_pred = svclassifier.predict(dev_data)
    sumnb = (dev_labels == dev_pred).sum()/len(dev_data)
    print(sumnb)

    gnb = GaussianNB()
    y_pred = gnb.fit(train_data, train_labels).predict(dev_data)
    sumnb = (dev_labels == y_pred).sum()/len(dev_data)
    print(sumnb)

    clf = LogisticRegression(solver='lbfgs', max_iter = 2000).fit(train_data, train_labels)
    pred_train = clf.predict(train_data)
    count = 0
    for i in range(len(pred_train)):
        if(pred_train[i]!=train_labels[i]):
            #print(i)
            count+=1
    #print(count/len(pred_train))

    count  = 0
    pred = clf.predict(dev_data)

    pred_test = gnb.predict(test_data)
    with open("output_rnn.csv",'w') as f:
        f.write("Id,Prediction\n")
        for i in range(len(pred_test)):
            f.write(test_ids[i])
            f.write(',%s\n' % pred_test[i])
            count+=1
    #print(pred_test)
    prob = clf.predict_proba(dev_data)
    #print(prob)

    score = clf.score(dev_data, dev_labels)
    print(score)

#main()

#readData("train.csv")