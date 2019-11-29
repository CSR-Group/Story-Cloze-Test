import csv
from nltk.tokenize import RegexpTokenizer

STORY = "STORY"
TRAIN_DATA_FILEPATH = "data/train.csv"
TEST_DATA_FILEPATH = "data/test.csv"
VALIDATION_DATA_FILEPATH = "data/dev.csv"
DEFAULT_TOKENIZER = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

class DataPoint:
    def __init__(self, inputSentences, nextSentence1, nextSentence2, answer):
        self.inputSentences = inputSentences
        self.nextSentence1 = nextSentence1
        self.nextSentence2 = nextSentence2
        self.answer = answer

def readFile(filename):
    data = []
    with open(filename) as csv_file:
        readFile = csv.DictReader(csv_file)
        for row in readFile:
            data.append(row)
    return data

def transform(data, tokenizer=DEFAULT_TOKENIZER, lower=True, tokenize=True):
    assert "InputSentence1" in data[0] 
    assert "InputSentence2" in data[0] 
    assert "InputSentence3" in data[0] 
    assert "InputSentence4" in data[0] 
    assert "RandomFifthSentenceQuiz1" in data[0] 
    assert "RandomFifthSentenceQuiz2" in data[0] 
    assert "AnswerRightEnding" in data[0]
    
    lowerTrans = lambda x: x if (lower == False) else x.lower()
    tokeniz = lambda x: tokenizer.tokenize(x) if tokenize else x 

    transformedData = []
    for dataPoint in data:
        InputSentence1 = tokeniz(lowerTrans(dataPoint["InputSentence1"]))
        InputSentence2 = tokeniz(lowerTrans(dataPoint["InputSentence2"]))
        InputSentence3 = tokeniz(lowerTrans(dataPoint["InputSentence3"]))
        InputSentence4 = tokeniz(lowerTrans(dataPoint["InputSentence4"]))
        nextSentence1 = tokeniz(lowerTrans(dataPoint["RandomFifthSentenceQuiz1"]))
        nextSentence2 = tokeniz(lowerTrans(dataPoint["RandomFifthSentenceQuiz2"]))
        answer = int(dataPoint["AnswerRightEnding"]) - 1
        transformedData.append(DataPoint([InputSentence1,InputSentence2,InputSentence3, InputSentence4], nextSentence1, nextSentence2, answer))        
    return transformedData

def getTrainData(filePath = TRAIN_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER, lower=True, tokenize=True):
    return transform(readFile(filePath), tokenizer, lower, tokenize)
    
def getTestData(filePath = TEST_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER, lower=True, tokenize=True):
    return transform(readFile(filePath), tokenizer, lower, tokenize)

def getValidationData(filePath = VALIDATION_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER, lower=True, tokenize=True):
    return transform(readFile(filePath), tokenizer, lower, tokenize)
