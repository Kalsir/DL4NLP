# Mostly copied from https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import tqdm

# Hyperparams
BATCH_SIZE = 8
MAX_LEN = 128
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
EPOCHS = 10
FINE = False

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
    )

    return {
      'questions': question,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
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

class QClassifier(nn.Module):
  def __init__(self, n_classes):
    super(QClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

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
    attention_mask = d["attention_mask"].to(device)
    targets = d["types"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
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
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      progress_bar.update()
  return correct_predictions.double() / n_examples, np.mean(losses)

# Load train and test data
train = pd.read_csv("train_1000.csv")
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

# Setup GPU
print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create model and move it to GPU
model = QClassifier(num_classes)
model = model.to(device)

# Setup optimizer/scheduler/loss function
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
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