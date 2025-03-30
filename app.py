import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import re
import string
import emoji

# Streamlit UI
st.title("Hate Speech Detection")

# File uploader for CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Load vocab (constant)
vocab = torch.load('vocab.pth')
PAD_IDX = vocab['<pad>']

# Define model
class Improved_BI_LSTM_GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text, lengths):
        embedded = F.dropout(self.embedding(text), p=0.2, training=self.training)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        attention_weights = F.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)
        context_vector = self.bn1(context_vector)
        return self.fc(context_vector)

# Load trained model
model = Improved_BI_LSTM_GloVe(vocab_size=len(vocab), embed_dim=100, hidden_dim=256, pad_idx=PAD_IDX, output_dim=6)
model.load_state_dict(torch.load('final.pth', map_location=torch.device('cpu')))
model.eval()

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = emoji.demojize(text)  # Convert emojis to text
    return text.lower().strip()

# Tokenization and Numericalization
tokenizer = get_tokenizer("basic_english")

def tokenize_and_numericalize(text, max_length=256):
    tokens = tokenizer(preprocess_text(text))
    numericalized = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens][:max_length]
    return torch.tensor(numericalized, dtype=torch.long)

# Process uploaded file
if uploaded_file is not None:
    st.write("Processing comments...")

    # Read dataset
    df = pd.read_csv(uploaded_file)
    
    # Rename column if necessary
    if 'text' in df.columns:
        df.rename(columns={'text': 'comment_text'}, inplace=True)

    # Process comments
    processed_comments = [tokenize_and_numericalize(comment) for comment in df["comment_text"]]
    lengths = torch.tensor([len(c) for c in processed_comments])

    # Pad sequences
    padded_comments = pad_sequence(processed_comments, batch_first=True, padding_value=PAD_IDX)

    # Predict
    with torch.no_grad():
        predictions = model(padded_comments, lengths).sigmoid().numpy()  # Convert logits to probabilities
        predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Add predictions to DataFrame
    df["toxic"] = predictions[:, 0]
    df["severe_toxic"] = predictions[:, 1]
    df["obscene"] = predictions[:, 2]
    df["threat"] = predictions[:, 3]
    df["insult"] = predictions[:, 4]
    df["identity_hate"] = predictions[:, 5]

    # Display results
    st.write("Prediction Results:")
    st.dataframe(df[["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])

    # Convert DataFrame to CSV and provide a download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download Predictions", data=csv, file_name="results.csv", mime="text/csv")
