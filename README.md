# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Theory
An LSTM-based model for recognizing named entities is a type of neural network that uses Long Short-Term Memory (LSTM) layers to identify and classify proper names and entities within a text, such as person names, locations, organizations, dates, etc. It is commonly employed in Named Entity Recognition (NER) tasks because LSTMs are effective at capturing sequential dependencies and context within text. Typically, these models process tokenized input data, learn contextual representations, and output labels for each token indicating whether it belongs to a specific entity type. This approach improves the accuracy of extracting meaningful information from unstructured text data.

## Problem Statement and Dataset

Develop a deep learning model using LSTM to perform Named Entity Recognition (NER) on text data. Preprocess the dataset by tokenizing text and encoding words and labels. Train and evaluate the model to accurately identify entities such as names, locations, and organizations.

<img width="741" height="864" alt="Screenshot 2026-03-24 141206" src="https://github.com/user-attachments/assets/4c56abf4-817b-406c-9212-b7f85dba8e53" />

## DESIGN STEPS

STEP 1:
Load data, create word/tag mappings, and group sentences.

STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

STEP 3:
Define dataset and DataLoader for batching.

STEP 4:
Build a bidirectional LSTM model for sequence tagging.

STEP 5:
Train the model over multiple epochs, tracking loss.

STEP 6:
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.


## PROGRAM

### Name: DIVYA LAKSHMI M

### Register Number: 212224040082

```python

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_train_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
      avg_train_loss = total_train_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      model.eval()
      total_val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch["labels"].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          total_val_loss += loss.item()
      avg_val_loss = total_val_loss / len(test_loader)
      val_losses.append(avg_val_loss)
      print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")


```

### OUTPUT

## Loss Vs Epoch Plot

<img width="670" height="568" alt="Screenshot 2026-03-16 084124" src="https://github.com/user-attachments/assets/18c36e88-b6d9-49ad-b2b1-0a6460d7aeaf" />

### Sample Text Prediction

<img width="380" height="501" alt="Screenshot 2026-03-16 084155" src="https://github.com/user-attachments/assets/47e4d32a-fac2-42fc-870b-1b7bcc59d0b4" />

## RESULT

Thus a LSTM-based model for recognizing the named entities in the text has been developed successfully.

