import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
import time
from tqdm import tqdm
from features1 import readData

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
	def __init__(self, input_dim, h):
			super(FFNN, self).__init__()
			self.h = h
			self.W1 = nn.Linear(input_dim, 16)
			self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
			self.W2 = nn.Linear(16, 4)	# BUGG - is 5
			self.W3 = nn.Linear(4, 1)
			# The below two lines are not a source for an error
			self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
			self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class
			self.sigmoid = nn.Sigmoid()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		# The z_i are just there to record intermediary computations for your clarity
		z1 = self.W1(input_vector)
		a1 = self.activation(z1)  # BUGG - not activated
		z2 = self.W2(a1)
		a2 = self.activation(z2)  # BUGG - not activated
		z3 = self.W3(a2)
		predicted_vector = self.sigmoid(z3)  # BUGG - double activation
		return predicted_vector

def main(hidden_dim = 25, number_of_epochs = 200):
	print("Fetching data")
	train_data, train_labels, train_ids = readData("train.csv","train4.csv")
	dev_data, dev_labels, dev_ids = readData("dev.csv","dev4.csv")
	test_data, test_labels, test_ids = readData("test.csv","predtest.csv")

	model = FFNN(input_dim = len(train_data[1]), h = hidden_dim)
	optimizer = optim.Adagrad(model.parameters(),lr=0.05)
	print("Training for {} epochs".format(number_of_epochs))
	for epoch in range(number_of_epochs):
		model.train()
		optimizer.zero_grad()
		criterion = nn.BCELoss()
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))
		
		c = list(zip(train_data, train_ids, train_labels))
		random.shuffle(c)
		train_data, train_ids, train_labels = zip(*c)
		
		minibatch_size = 16 
		N = len(train_data)   
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector = train_data[minibatch_index * minibatch_size + example_index]
				gold_label = train_labels[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(torch.FloatTensor(input_vector))

				if predicted_vector > 0.5:
					predicted_label = 1
				else:
					predicted_label = 0

				#predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == int(gold_label))
				total += 1
				#example_loss = model.compute_Loss(predicted_vector, torch.tensor([gold_label]))
				example_loss = criterion(predicted_vector, torch.tensor(float(gold_label)))
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size	# BUGG - not averaging loss
			loss.backward()		# BUGGGG - loss and optimzer updated once per epoch
			optimizer.step()
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch + 1))
		
		
		c = list(zip(dev_data, dev_ids, dev_labels))
		random.shuffle(c)
		dev_data, dev_ids, dev_labels = zip(*c)

		minibatch_size = 16 
		N = len(dev_data) 
		for minibatch_index in tqdm(range(N // minibatch_size)):
			# optimizer.zero_grad()
			# loss = None
			# BUGGGG - Shouldnt train on validation set
			for example_index in range(minibatch_size):
				input_vector = dev_data[minibatch_index * minibatch_size + example_index]
				gold_label = dev_labels[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(torch.FloatTensor(input_vector))
				
				if predicted_vector > 0.5:
					predicted_label = 1
				else:
					predicted_label = 0

				correct += int(predicted_label == int(gold_label))
				total += 1
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))
		
main()