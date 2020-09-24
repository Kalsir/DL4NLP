# BERT parts mostly copied from https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import tqdm
from better_lstm import LSTM

# Hyperparams
BATCH_SIZE = 8
MAX_LEN = 128
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
EPOCHS = 5
FINE = False
MODEL = 'BERT' #Bert or LSTM
UNCERTAINTY_PASSES = 50
TRAINING_FILE = "train_1000.csv"
OUTPUTFILE = "bert_1000_drop_0.3.txt"
BERT_DROPOUT = 0.3
LSTM_DROPOUT = 0.25
TYPES = ["HUM", "LOC", "ENTY", "ABBR", "DESC", "NUM"]

# Open file and write some info
f = open(OUTPUTFILE, "w")
f.write("Hyperparameters\n")
f.write("Model: " + MODEL + "\n")
f.write("Maximum sentence length: " + str(MAX_LEN) + "\n")
f.write("Fine types: " + str(FINE) + "\n")
f.write("Uncertainty passes: " + str(UNCERTAINTY_PASSES) + "\n")
f.write("Epochs: " + str(EPOCHS) + "\n")
f.write("Training file: " + TRAINING_FILE + "\n")
f.write("Batch size: " + str(BATCH_SIZE) + "\n")
if MODEL == 'BERT':
	f.write("Dropout: " + str(BERT_DROPOUT) + "\n")
else:
	f.write("Dropoutw: " + str(LSTM_DROPOUT) + "\n\n")
	f.write("Dropouti: " + str(LSTM_DROPOUT) + "\n\n")
	f.write("Dropouto: " + str(LSTM_DROPOUT) + "\n\n")

# Some utility classes/functions
class QCDataset(Dataset):
  def __init__(self, questions, targets, tokenizer, max_len):
    self.questions = questions
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):

    return len(self.questions)

  def __getitem__(self, item):
    question = str(self.questions[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      question,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      return_attention_mask=True,
      return_tensors='pt',
      return_length = True
    )

    return {
      'questions': question,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'text_lengths' : encoding['length'].flatten().squeeze(),
      'types': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size, fine = False):
  if fine:
  	targets = df.qtype.to_numpy()
  else:
  	targets = df.qtype_coarse.to_numpy()
  ds = QCDataset(
    questions=df.question.to_numpy(),
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
  )

class BertClassifier(nn.Module):
  def __init__(self, n_classes):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=BERT_DROPOUT)
    self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.fc(output)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, embedding_dim = 768, hidden_dim = 768, n_layers = 2, 
                 bidirectional = True, dropouti = LSTM_DROPOUT, dropouto = LSTM_DROPOUT, dropoutw = LSTM_DROPOUT, unit_forget_bias=True):
        super(LSTMClassifier, self).__init__()          
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropouti = dropouti,
                           dropouto = dropouto,
                           dropoutw = dropoutw,
                           unit_forget_bias=unit_forget_bias,
                           batch_first=True)
        if bidirectional:
        	self.fc = nn.Linear(hidden_dim * 2, n_classes)
        else:
        	self.fc = nn.Linear(hidden_dim, n_classes)
        self.bidirectional = bidirectional
        
    def forward(self, input_ids, text_lengths):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.bidirectional:
        	hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
        	hidden = hidden[-1,:,:]
        return self.fc(hidden)   

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0

  progress_bar = tqdm.tqdm(total=len(data_loader), desc='Batches')
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    targets = d["types"].to(device)
    if MODEL == 'LSTM':
    	text_lengths = d["text_lengths"].to(device)
    	outputs = model(input_ids, text_lengths)
    else:
    	attention_mask = d["attention_mask"].to(device)
    	outputs = model(input_ids, attention_mask)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    progress_bar.update()
  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_uncertainty(model, data_loader, loss_fn, device, n_examples, label_encoder):
	model = model.train()
	progress_bar = tqdm.tqdm(total=len(data_loader), desc='Test samples')
	uncertainty_difference = 0
	uncertainty_difference_success = 0
	uncertainty_difference_fail = 0
	counter = 0
	fail_counter = 0
	precisions = [0, 0, 0, 0, 0, 0]
	prediction_counters = [0, 0, 0, 0, 0, 0]
	actual_counters = [0, 0, 0, 0, 0, 0]
	true_positive_counters = [0, 0, 0, 0, 0, 0]
	certainties = [0, 0, 0, 0, 0, 0]
	certainties_ = [0, 0, 0, 0, 0, 0]
	recalls = [0, 0, 0, 0, 0, 0]
	for d in data_loader:
		input_ids = d["input_ids"].to(device)
		attention_mask = d["attention_mask"].to(device)
		targets = d["types"].to(device)
		if MODEL == 'LSTM':
			secondary =d["text_lengths"].to(device)
		else:
			secondary = d["attention_mask"].to(device)
			outputs = model(input_ids, attention_mask)
		pred_list = []
		for i in range(UNCERTAINTY_PASSES):
			outputs = model(input_ids, secondary)
			_, pred = torch.max(outputs, dim=1)
			pred_list.append(pred)
		unique, counts = torch.unique(torch.stack(pred_list), return_counts = True)
		most_frequent = counts.argmax()
		prediction = unique[most_frequent]
		certainty = counts[most_frequent].item()/UNCERTAINTY_PASSES
		predicted_type = label_encoder.inverse_transform([prediction.cpu().item()])[0]
		actual_type = label_encoder.inverse_transform([d["types"].item()])[0]
		if predicted_type != actual_type:
			f.write("Question:" + d["questions"][0])
			f.write("Predicted Type:" + label_encoder.inverse_transform([prediction.cpu().item()])[0] + "\n")
			f.write("Actual Type:" + label_encoder.inverse_transform([d["types"].item()])[0] + "\n")
			f.write("Certainty:" + str(certainty) + "\n")
			uncertainty_difference +=  certainty
			uncertainty_difference_fail +=  certainty
			fail_counter += 1
		else:
			uncertainty_difference += 1-certainty
			uncertainty_difference_success += 1-certainty
		counter += 1
		for i, t in enumerate(TYPES):
			if predicted_type == t:
				prediction_counters[i] += 1
				if predicted_type == actual_type:
					true_positive_counters[i] += 1
				certainties[i] += certainty
			if actual_type == t:
				actual_counters[i] += 1
		progress_bar.update()
	for i in range(6):
		if prediction_counters[i] == 0:
			precisions[i] = -1
			certainties_[i] = -1
		else:
			precisions[i] = (true_positive_counters[i]/prediction_counters[i])
			certainties_[i] = certainties[i]/prediction_counters[i]
		recalls[i] = true_positive_counters[i]/actual_counters[i]
		f.write(TYPES[i] + " precision: " + str(precisions[i]) + " average certainty: " + str(certainties_[i]) +  " # of times predicted: " + str(prediction_counters[i]) + "\n"
			+  " # of times actual: " + str(actual_counters[i]) + " recall: " + str(recalls[i]) + "\n")
	uncertainty_difference = uncertainty_difference/counter
	uncertainty_difference_fail = uncertainty_difference_fail/fail_counter
	uncertainty_difference_success = uncertainty_difference_success/(counter - fail_counter)
	f.write("Uncertainty difference: " + str(uncertainty_difference) + "\n")
	f.write("Uncertainty difference for fail: " + str(uncertainty_difference_fail) + "\n")
	f.write("Uncertainty difference for success: " + str(uncertainty_difference_success) + "\n")
	f.write("Average precision (every class equal): " + str(sum(precisions)/6) + "\n")
	f.write("Average recall (every class equal): " + str(sum(recalls)/6) + "\n")
	f.write("Average certainty (every class equal): " + str(sum(certainties_)/6) + "\n")
	f.write("Average certainty: " + str(sum(certainties)/500) + "\n")
	f.write("Accuracy: " + str(sum(true_positive_counters)/sum(prediction_counters)) + "\n")

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    progress_bar = tqdm.tqdm(total=len(data_loader), desc='Batches')
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["types"].to(device)
      if MODEL == 'LSTM':
      	text_lengths =d["text_lengths"].to(device)
      	outputs = model(input_ids, text_lengths)
      else:
      	attention_mask = d["attention_mask"].to(device)
      	outputs = model(input_ids, attention_mask)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      progress_bar.update()
  return correct_predictions.double() / n_examples, np.mean(losses)

# Load train and test data
train = pd.read_csv(TRAINING_FILE)
test = pd.read_csv("test_500.csv")

# Encode question types as numbers
le = LabelEncoder()
le.fit(pd.Series(train.qtype.tolist() + test.qtype.tolist()).values)
train['qtype'] = le.transform(train.qtype.values)
test['qtype'] = le.transform(test.qtype.values)
le2 = LabelEncoder()
le2.fit(pd.Series(train['qtype_coarse'].tolist() + test['qtype_coarse'].tolist()).values)
train['qtype_coarse'] = le2.transform(train['qtype_coarse'].values)
test['qtype_coarse'] = le2.transform(test['qtype_coarse'].values)
num_types = len(le.classes_)
num_coarse = len(le2.classes_)
if FINE:
	num_classes = num_types
else:
	num_classes = num_coarse

# Get tokenizer for preprocessing
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# Create dataloaders
train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE, FINE)
test_data_loader = create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE, FINE)
test_uncertainty_data_loader = create_data_loader(test, tokenizer, MAX_LEN, 1, FINE)

# Setup GPU
print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create model and move it to GPU
if MODEL == 'LSTM':
	model = LSTMClassifier(tokenizer.vocab_size, num_classes)
else:
	model = BertClassifier(num_classes)
model = model.to(device)

# Setup optimizer/scheduler/loss function
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False, weight_decay = 0.001) # Added 0.0001 weight decay as suggested in A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

# Train model
print("Number of classes", num_classes)
print("Batch size", BATCH_SIZE)
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  f.write(f'Train loss {train_loss} accuracy {train_acc}' +"\n\n")

# Test model
print('Testing model')
print('-' * 10)
test_acc, test_loss = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(test)
  )
print(f'Test loss {test_loss} accuracy {test_acc}')
f.write(f'Test loss {test_loss} accuracy {test_acc}' +"\n\n")

# Evaluate uncertainty
print('Evaluating uncertainty')
print('-' * 10)
eval_uncertainty(
    model,
    test_uncertainty_data_loader,
    loss_fn,
    device,
    len(test),
    le2
  )