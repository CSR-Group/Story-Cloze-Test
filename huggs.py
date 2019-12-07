import ingest
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForNextSentencePrediction
from tqdm import tqdm, trange
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random


MAX_LEN = 90
BATCH_SIZE = 50
tokenizer = BertTokenizer.from_pretrained('data/mini/vocab.txt', do_lower_case=True)
model = BertForNextSentencePrediction.from_pretrained('data/mini/')


def tokenize(sentence):
    # tokenize the paragraph
    sentence = tokenizer.tokenize(sentence)
    # convert each token to its vocab id
    sentence = tokenizer.convert_tokens_to_ids(sentence)
    return sentence

def transform(data, returnAsPair=False):
    transformedData = []
    for dataPoint in data:
        # prestory = ["[CLS] " + query + " [SEP]" for query in dataPoint.inputSentences]
        prestory = " ".join(dataPoint.inputSentences)
        prestory = "[CLS] " + prestory + " [SEP] "
        
        prestory = tokenize(prestory)
        end1 = tokenize(dataPoint.nextSentence1 + " [SEP]")
        end2 = tokenize(dataPoint.nextSentence2 + " [SEP]")
        
        prestorySegments = [0 for token in prestory]
        end1Segments = [1 for token in end1]
        end2Segments = [1 for token in end2]
        story1Seg = prestorySegments + end1Segments
        story2Seg = prestorySegments + end2Segments

        story1 = prestory + end1
        story2 = prestory + end2
        assert len(story1) == len(story1Seg)
        assert len(story2) == len(story2Seg)
        # pad the sequence of tokens
        story1 = pad_sequences([story1], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]
        story2 = pad_sequences([story2], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]
        segmentMask1 = pad_sequences([story1Seg], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]
        segmentMask2 = pad_sequences([story2Seg], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]
        attention1 = [float(i>0) for i in story1]
        attention2 = [float(i>0) for i in story2]

        story1 = torch.tensor([story1])
        segmentMask1 = torch.tensor([segmentMask1])
        attention1 = torch.tensor([attention1])
        story2 = torch.tensor([story2])
        segmentMask2 = torch.tensor([segmentMask2])
        attention2 = torch.tensor([attention2])

        if(returnAsPair):
            transformedData.append(({"story" : story1,"segmentMask" : segmentMask1,"attention": attention1, "storyRaw" : dataPoint.inputSentences + [dataPoint.nextSentence1]}, 
                                    {"story" : story2,"segmentMask" : segmentMask2,"attention": attention2, "storyRaw" : dataPoint.inputSentences + [dataPoint.nextSentence2]},
                                    dataPoint.answer))
        else:            
            transformedData.append({"story" : story1,"segmentMask" : segmentMask1,"attention": attention1, "label":dataPoint.answer == 0})
            transformedData.append({"story" : story2,"segmentMask" : segmentMask2,"attention": attention2, "label":dataPoint.answer == 1})
            
    return transformedData
    
def main():
    rawdata = ingest.getTrainData(tokenize=False, lower=False)
    validData = ingest.getValidationData(tokenize=False, lower=False)
    transformedDataSet = transform(rawdata)
    transformedValidationDataSet = transform(validData)
    
    optimizer = BertAdam(model.parameters(),lr=0.01,warmup=.1)

    epochs = 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        print("Training started for epoch {}".format(epoch + 1))

        correct = 0
        total = 0
        tr_loss = 0

        random.shuffle(transformedDataSet)
        dataSet = np.array_split(transformedDataSet, BATCH_SIZE)
        trainSet = []
        for data in dataSet:
            story = []
            segmentMask = []
            attention = []
            label = []
            for dataPoint in data:
                story.append(dataPoint["story"])
                segmentMask.append(dataPoint["segmentMask"])
                attention.append(dataPoint["attention"])
                label.append(torch.LongTensor([0 if dataPoint["label"] else 1]))
            trainSet.append((torch.cat(story),torch.cat(segmentMask),torch.cat(attention),torch.cat(label)))            
        
        for dataPoint in tqdm(trainSet):
            story = dataPoint[0]
            segmentMask = dataPoint[1]
            attention = dataPoint[2]
            label = dataPoint[3]

            # Forward pass
            seq_relationship_score = model(story, token_type_ids=segmentMask, attention_mask = attention)
            
            for i in range(len(seq_relationship_score)):
                if(label[i] == 0 and (seq_relationship_score[i][0] > seq_relationship_score[i][1])):
                    correct += 1
                elif(label[i] == 1 and (seq_relationship_score[i][0] < seq_relationship_score[i][1])):
                    correct += 1
                total += 1
            
            # Backward pass
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(seq_relationship_score.view(-1, 2), label.view(-1))
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training loss for epoch {}: {}".format(epoch + 1, tr_loss / total))

        correct = 0
        total = 0

        for dataPoint in tqdm(transformedValidationDataSet):
            story = dataPoint["story"]
            segmentMask = dataPoint["segmentMask"]
            attention = dataPoint["attention"]
            label = torch.LongTensor([0 if dataPoint["label"] else 1])
            # Forward pass
            seq_relationship_score = model(story, token_type_ids=segmentMask, attention_mask = attention)
            if(dataPoint["label"] and (seq_relationship_score[0][0] > seq_relationship_score[0][1])):
                correct += 1
            elif(not dataPoint["label"] and (seq_relationship_score[0][0] < seq_relationship_score[0][1])):
                correct += 1
            total += 1

        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))

    # path = "trainedModel_101"
    # torch.save(model.state_dict(),path + str(epoch + 1) + ".pth")

            
if __name__ == "__main__":
    main()
