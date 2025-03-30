import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import spacy
import re
import string
import emoji
from nltk.corpus import stopwords
import nltk

# Download necessary resources
nltk.download('stopwords')

# Streamlit UI
st.title("Automatic Content Moderator Helper")

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

# Load NLP utilities
nlp = spacy.load("en_core_web_sm")
stpwds = set(stopwords.words('english'))

# Define preprocessing patterns
punc = string.punctuation.replace('#', '').replace('!', '').replace('?', '') + "∞θ÷α•à−β∅³π‘₹´°£€\\×™√²—"
patterns = [
    r'\\[nrtbfv\\]',         # \n, \t etc
    '<.*?>',                 # HTML tags
    r'https?://\S+|www\.\S+', # Links
    r'\ufeff',               # BOM characters
    r'^[^a-zA-Z0-9]+$',      # Non-alphanumeric tokens
    r'ｗｗｗ．\S+',            # Full-width URLs
    r'[\uf700-\uf7ff]',      # Unicode private-use chars
    r'^[－—…]+$',            # Special punctuation
    r'[︵︶]'                # CJK parentheses
]

# Chat words mapping
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    # ... include all your chat words mapping
}

time_zone_abbreviations = {
    "UTC", "GMT", "EST", "CST", "PST", "MST",
    "EDT", "CDT", "PDT", "MDT", "CET", "EET",
    "WET", "AEST", "ACST", "AWST", "HST",
    "AKST", "IST", "JST", "KST", "NZST"
}

def preprocess_text(text):
    """Apply all preprocessing steps to a single text"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Apply regex patterns
    for regex in patterns:
        text = re.sub(regex, '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans(punc, ' ' * len(punc)))
    
    # Remove time zones and stopwords
    text = ' '.join(word for word in text.split() 
                   if word not in time_zone_abbreviations 
                   and word not in stpwds)
    
    # Expand chat words
    text = ' '.join(chat_words.get(word.lower(), word) for word in text.split())
    
    # Lowercase and emoji handling
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
    filtered_comments = [c for c in processed_comments if len(c) > 0]  # Remove empty sequences

    if len(filtered_comments) == 0:
        st.error("No valid comments found after preprocessing. Check your dataset.")
    else:
        lengths = torch.tensor([len(c) for c in filtered_comments])
        padded_comments = pad_sequence(filtered_comments, batch_first=True, padding_value=PAD_IDX)

        # Predict
        with torch.no_grad():
            predictions = model(padded_comments, lengths).sigmoid().numpy()
            predictions = (predictions > 0.5).astype(int)

        # Add predictions back to DataFrame
        df = df[df["comment_text"].apply(lambda x: len(tokenize_and_numericalize(x)) > 0)]  # Keep only non-empty
        df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predictions

        # Display flagged comments
        flagged_df = df[(df[["toxic", "severe_toxic", "obscene", "threat"]] == 1).any(axis=1)]
        st.write("The following comments have been flagged:")
        st.dataframe(flagged_df[["comment_text", "toxic", "severe_toxic", "obscene", "threat"]])
        
        # Provide download button
        csv = flagged_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Flagged Comments", data=csv, file_name="flagged_comments.csv", mime="text/csv")
