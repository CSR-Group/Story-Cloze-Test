from ingest import *
import nltk
from nltk.corpus import wordnet as wn


def getPosTags(sentence):
    return nltk.pos_tag(sentence)

def getEntities(sentence):
    nouns = set()
    entity_type = {'NNPS', 'NNS', 'NNP', 'PRP', 'NN', 'PRP$'}
    taggedWords = getPosTags(sentence)
    for (x,y) in taggedWords:
        if(y in entity_type):
            nouns.add(x)
    return nouns

def getAllEntiries(sentences):
    nouns = set()
    for sentence in sentences:
        nouns.update(getEntities(sentence))
    return nouns

def getTags(sentence):
    entity_type = {'NNPS', 'NNS', 'NNP', 'PRP', 'NN', 'PRP$'}
    verb_type = {'VB','VBD','VBG','VBN','VBP','VBZ'}
    adv_type = {'RB','RBR','RBS'}
    adj_type = {'JJ', 'JJR', 'JJS'}
    taggedWords = getPosTags(sentence)

    nouns = set()
    verbs = set()
    adj = set() 
    adv = set()

    for (x,y) in taggedWords:
        if(y in entity_type): 
            nouns.add(x)
        if(y in verb_type): 
            verbs.add(x)
        if(y in adv_type): 
            adv.add(x)
        if(y in adj_type): 
            adj.add(x)
    return {'N':nouns,'V':verbs, 'ADV':adv, 'ADJ':adj}

if __name__ == "__main__":
    data = getTrainData()
    for datapoint in data[5:6]:
        print(datapoint.inputSentences)
        print(datapoint.nextSentence1)
        print(datapoint.nextSentence2)
        print(getPosTags(datapoint.inputSentences[0]))
        print(getPosTags(datapoint.inputSentences[1]))
        print(getPosTags(datapoint.inputSentences[2]))
        print(getPosTags(datapoint.inputSentences[3]))
    
        print(getPosTags(datapoint.nextSentence1))
        print(getPosTags(datapoint.nextSentence2))
    
        # print(str(getAllEntiries(datapoint.inputSentences)) + " - " + str(getEntities(datapoint.nextSentence1)) + " - " + str(getEntities(datapoint.nextSentence1)))
    