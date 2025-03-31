import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import requests
import spacy
import re
import string
import emoji
from nltk.corpus import stopwords
import nltk

# Download necessary resources
nltk.download('stopwords')

# Streamlit UI
st.title("Instagram Comment Scraper & Hate Speech Detection")

# Input for Instagram Post URL
post_url = st.text_input("Enter Instagram Post URL")

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
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
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

# Function to call the API and fetch comments
def fetch_instagram_comments(post_url):
    api_url = "http://localhost:3001/api/scrape-comments"
    payload = {"postUrl": post_url}
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("success") and "comments" in data:
            return [comment["text"] for comment in data["comments"]]
        else:
            st.error("No comments found or API returned an error.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return []

# Process Instagram comments
if st.button("Scrape and Analyze Comments") and post_url:
    st.write("Fetching comments...")
    
    comments = fetch_instagram_comments(post_url)
    
    if comments:
        df = pd.DataFrame(comments, columns=['comment_text'])
        
        processed_comments = [tokenize_and_numericalize(comment) for comment in df["comment_text"]]
        filtered_comments = [c for c in processed_comments if len(c) > 0]
        
        if filtered_comments:
            lengths = torch.tensor([len(c) for c in filtered_comments])
            padded_comments = pad_sequence(filtered_comments, batch_first=True, padding_value=PAD_IDX)
            
            with torch.no_grad():
                predictions = model(padded_comments, lengths).sigmoid().numpy()
                predictions = (predictions > 0.5).astype(int)
            
            df = df[df["comment_text"].apply(lambda x: len(tokenize_and_numericalize(x)) > 0)]
            df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predictions
            
            flagged_df = df[(df[["toxic", "severe_toxic", "obscene", "threat"]] == 1).any(axis=1)]
            
            st.write("Flagged comments:")
            st.dataframe(flagged_df[["comment_text", "toxic", "severe_toxic", "obscene", "threat"]])
            
            csv = flagged_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Flagged Comments", data=csv, file_name="flagged_comments.csv", mime="text/csv")
        else:
            st.error("No valid comments found after preprocessing.")
    else:
        st.error("No comments found on the post.")
