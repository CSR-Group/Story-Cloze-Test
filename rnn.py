import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import pickle
import time
import vsmlib
from preprocess import *
from torch.nn import init
from tqdm import tqdm
from torch.autograd import Variable
import  matplotlib.pyplot as plt
import csv
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_dim = 500
        self.hidden_dim = 64
        self.output_dim = 1
        self.num_rnn_layers = 2
        self.nonlinearity = 'tanh'
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        self.lstm = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True)
        self.rnn = nn.RNN(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, nonlinearity=self.nonlinearity)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def get_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.num_rnn_layers, inputs.size(0), self.hidden_dim))
        #out, hn = self.rnn(inputs, h0)
        out, hn = self.lstm(inputs, h0)
        z1 = self.fc(out[:, -1, :])
        return self.sigmoid(z1)
 
def performTrain(model, optimizer, train_data, train_ids):
    c = list(zip(train_data, train_ids))
    random.shuffle(c)
    train_data, train_ids = zip(*c)

    #random.shuffle(train_data)
    predicted_prob = []
    gold_labels = []
    N = len(train_data)
    correct = 0
    total = 0
    totalloss = 0
    minibatch_size = 6
    criterion = nn.BCELoss()

    for minibatch_index in tqdm(range(N // minibatch_size)):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
            input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
            predicted_vector = model(input_vector.float())
            predicted_prob.append(predicted_vector)
            gold_labels.append(gold_label)
            #predicted_label = torch.argmax(predicted_vector)
            if predicted_vector > 0.5:
                predicted_label = 1
            else:
                predicted_label = 0
            correct += int(predicted_label == gold_label)
            total +=1
            #instance_loss = model.get_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
            predicted_vector = predicted_vector.squeeze(1)
            predicted_vector = predicted_vector.squeeze(0)
            l = criterion(predicted_vector, torch.tensor(float(gold_label)))
            if(loss is None):
                loss = l
            else:
                loss += l
        loss = loss / minibatch_size
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        totalloss +=loss
    accuracy = (correct / total) * 100
    return totalloss/(N // minibatch_size), accuracy, predicted_prob, gold_labels, train_ids

def validate(model, val_data):
    criterion = nn.BCELoss()
    correct = 0
    loss = None
    true_label = []
    pred_label = []
    pred_prob = []
    for i in tqdm(range(len(val_data))):
        input_vector, gold_label = val_data[i]
        true_label.append(gold_label)
        predicted_vector = model(input_vector.float())
        pred_prob.append(predicted_vector)
        #predicted_label = torch.argmax(predicted_vector)
        if predicted_vector > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        pred_label.append(predicted_label)

        correct += int(predicted_label == gold_label)
        #instance_loss = model.get_loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
        predicted_vector = predicted_vector.squeeze(1)
        predicted_vector = predicted_vector.squeeze(0)
        l = criterion(predicted_vector, torch.tensor(float(gold_label)))

        if(loss is None):
            loss = l
        else:
            loss += l
    loss = loss / len(val_data)
    accuracy = (correct / len(val_data)) * 100
    fscore  = f1_score(true_label,pred_label)
    recall = recall_score(true_label,pred_label)
    precision = precision_score(true_label,pred_label)

    target_names = ['class 0', 'class 1']
    print(classification_report(true_label, pred_label, target_names=target_names))
    #print(pred_label)
    return loss.data, accuracy, fscore, recall, precision, pred_prob

def test(model, test_data):
    pred_label = []
    pred_prob = []
    for i in tqdm(range(len(test_data))):
        input_vector, _ = test_data[i]
        predicted_vector = model(input_vector.float())
        pred_prob.append(predicted_vector)
        if predicted_vector > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        pred_label.append(predicted_label)
    return pred_prob, pred_label

def main(num_epoch = 15):
    count = 15
    train_ids, train_data = readData("train.csv")
    dev_ids, dev_data = readData("dev.csv")
    model = RNN()
    optimizer = optim.Adagrad(model.parameters(),lr=0.001)
    model = model.float()
    criterion = nn.BCELoss()

    model.load_state_dict(torch.load("GRUmodel15.pth"))

    if os.path.exists("rnnmodel.pth"):
        model.load_state_dict(torch.load("rnnmodel.pth"))
        print("Successful")
        res = []
        N = len(val_data)
        for i in range(N):
            input_vector, gold_label = val_data[i]
            predicted_vector = model(input_vector.float())
            predicted_label = torch.argmax(predicted_vector)
            if predicted_label != gold_label:
                temp = []
                temp.append(i)
                temp.append(predicted_label)
                temp.append(gold_label)
                print(i)
                print(predicted_label)
                print(gold_label)
                res.append(temp)
        with open('rnn_64_2.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(res)
        csvFile.close()
        return

    train_accuracy_history = []
    train_loss_history = []

    val_loss_history = []
    val_accuracy_history = []
    val_fscore_history = []
    val_recall_history = []
    val_precision_history = []

    for epoch in range(num_epoch):

        # if os.path.exists("rnnmodel.pth"):
        #     state_dict = torch.load("model.pth")['state_dict']
        #     model.load_state_dict(state_dict)
        #     print("Successful")

        # if len(train_loss_history)>1 and (train_loss_history[-1] < val_loss_history[-1]) and (train_loss_history[-1] < train_loss_history[-2]) and (val_loss_history[-1] > val_loss_history[-2]):
        #     break
        
        count += 1
        model.train()
        optimizer.zero_grad()
        start_time = time.time()
        train_loss, train_accuracy, train_predicted_prob, train_gold_labels, ids = performTrain(model, optimizer, train_data, train_ids)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        start_time = time.time()
        val_loss, val_accuracy, val_fscore, val_recall, val_precision,  val_predicted_prob = validate(model, dev_data)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_accuracy))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
        val_fscore_history.append(val_fscore)
        val_recall_history.append(val_recall)
        val_precision_history.append(val_precision)

        #saving model aftr every epoch
        path = "GRUmodel"
        torch.save(model.state_dict(),path + str(count) + ".pth")

        #save the predicted rnn prob feature
        train_rnn_pred = []
        dev_rnn_pred = []
        for i in range(len(train_ids)):
            _, label = train_data[i]
            for j in range(len(ids)):
                if ids[j] == train_ids[i] and label == train_gold_labels[j]:
                    train_rnn_pred.append(train_predicted_prob[j])
        
        for i in range(len(dev_ids)):
            dev_rnn_pred.append(val_predicted_prob[i])

        file_path = "GRUtrain" + str(count) +".csv"
        with open(file_path, "w") as f:
            for i in range(len(train_rnn_pred)):
                f.write(train_ids[i])
                _,label = train_data[i]
                f.write(',%s' % label)
                f.write(',%s\n' % train_rnn_pred[i])

        file_path = "GRUdev" + str(count) +".csv"
        with open(file_path, "w") as f:
            for i in range(len(dev_rnn_pred)):
                f.write(dev_ids[i])
                _,label = dev_data[i]
                f.write(',%s' % label)
                f.write(',%s\n' % dev_rnn_pred[i])

    print("Training Set Metrics")
    print(train_accuracy_history)
    print(train_loss_history)

    print("Validation Set Metrics")
    print(val_loss_history)
    print(val_accuracy_history)
    print(val_fscore_history)
    print(val_precision_history)
    print(val_recall_history)

    print("Number of Parameters")
    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        if p.requires_grad:
            print(p.numel())
    print(pytorch_total_params)

    # training loss 
    iteration_list = [i+1 for i in range(count)]
    plt.plot(iteration_list,train_loss_history)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('train_loss_history.png')
    plt.clf()
    
    # training accuracy
    plt.plot(iteration_list,train_accuracy_history)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('train_accuracy_history.png')
    plt.clf()

    # validation loss 
    plt.plot(iteration_list,val_loss_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('val_loss_history.png')
    plt.clf()

    # validation accuracy
    plt.plot(iteration_list,val_accuracy_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('val_accuracy_history.png')
    plt.clf()

    # validation fscore
    plt.plot(iteration_list,val_fscore_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation FScore")
    plt.title("RNN: Fscore vs Number of Epochs")
    #plt.show()
    plt.savefig('val_fscore_history.png')
    plt.clf()

    # validation recall
    plt.plot(iteration_list,val_recall_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Recall")
    plt.title("RNN: Recall vs Number of Epochs")
    #plt.show()
    plt.savefig('val_recall_history.png')
    plt.clf()

    # validation precision
    plt.plot(iteration_list,val_precision_history,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Precision")
    plt.title("RNN: Precision vs Number of Epochs")
    #plt.show()
    plt.savefig('val_precision_history.png')
    plt.clf()

def predict_test(pathname):
    model = RNN()
    optimizer = optim.Adagrad(model.parameters(),lr=0.01)
    model = model.float()
    if os.path.exists(pathname):
        model.load_state_dict(torch.load(pathname))
        print("Successful")

    test_ids, test_data  = readData("test.csv")
    pred_prob, pred_label = test(model,test_data)

    file_path = "predtest.csv"
    with open(file_path, "w") as f:
        for i in range(len(pred_prob)):
            f.write('%s\n' % pred_prob[i])

#main()
predict_test("GRUmodel17.pth")