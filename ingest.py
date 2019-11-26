import csv
from nltk.tokenize import RegexpTokenizer


STORY = "STORY"
TRAIN_DATA_FILEPATH = "data/dev.csv"
TEST_DATA_FILEPATH = "data/test.csv"
VALIDATION_DATA_FILEPATH = "data/dev.csv"
DEFAULT_TOKENIZER = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

def readFile(filename):
    data = []
    with open(filename) as csv_file:
        readFile = csv.DictReader(csv_file)
        for row in readFile:
            data.append(row)
    return data

def transform(data, tokenizer=DEFAULT_TOKENIZER):
    assert "InputSentence1" in data[0] 
    assert "InputSentence2" in data[0] 
    assert "InputSentence3" in data[0] 
    assert "InputSentence4" in data[0] 
    assert "RandomFifthSentenceQuiz1" in data[0] 
    assert "RandomFifthSentenceQuiz2" in data[0] 
    assert "AnswerRightEnding" in data[0]

    for dataPoint in data:
        dataPoint["InputSentence1"] = tokenizer.tokenize(dataPoint["InputSentence1"])
        dataPoint["InputSentence2"] = tokenizer.tokenize(dataPoint["InputSentence2"])
        dataPoint["InputSentence3"] = tokenizer.tokenize(dataPoint["InputSentence3"])
        dataPoint["InputSentence4"] = tokenizer.tokenize(dataPoint["InputSentence4"])
        dataPoint["RandomFifthSentenceQuiz1"] = tokenizer.tokenize(dataPoint["RandomFifthSentenceQuiz1"])
        dataPoint["RandomFifthSentenceQuiz2"] = tokenizer.tokenize(dataPoint["RandomFifthSentenceQuiz2"])
        dataPoint["AnswerRightEnding"] = int(dataPoint["AnswerRightEnding"]) - 1
    return data

def getTrainData(filePath = TRAIN_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER):
    return transform(readFile(filePath), tokenizer)
    
def getTestData(filePath = TEST_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER):
    return transform(readFile(filePath), tokenizer)

def getValidationData(filePath = VALIDATION_DATA_FILEPATH, tokenizer=DEFAULT_TOKENIZER):
    return transform(readFile(filePath), tokenizer)
    
print(getValidationData())