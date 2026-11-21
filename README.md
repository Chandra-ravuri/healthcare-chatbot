# Healthcare Chatbot (Python + Keras)

A simple intent-based chatbot for a healthcare app. You type a request in plain English
(e.g. *"find me a pharmacy"* or *"show blood pressure results for patient"*), a small neural
network works out which topic you mean, and the bot replies with a matching response.

Based on [tridibsamanta/Chatbot-using-Python](https://github.com/tridibsamanta/Chatbot-using-Python),
updated to work with current TensorFlow/Keras 3 and NumPy 2.

## What it understands

| Intent | Example |
|---|---|
| greeting | "Hello", "How are you" |
| goodbye | "Bye", "See you later" |
| thanks | "Thank you" |
| options | "What can you do?" |
| adverse_drug | "Open adverse drugs module" |
| blood_pressure | "I want to log blood pressure results" |
| blood_pressure_search | "Show blood pressure results for patient" |
| pharmacy_search | "Find me a pharmacy" |
| hospital_search | "Lookup for hospital" |

## Project files

| File | Purpose |
|---|---|
| `intents.json` | Topics, example sentences and replies |
| `train_chatbot.py` | Trains the neural network on `intents.json` |
| `chatgui.py` | Tkinter chat window |
| `chatbot_model.keras`, `words.pkl`, `classes.pkl` | Trained model files (created by training) |

## How it works

1. **Training:** each example sentence is tokenized and lemmatized with NLTK, then turned into a
   bag-of-words vector. A Keras network (128 → 64 → softmax) learns to map vectors to intents.
2. **Chatting:** your message is converted the same way, the model predicts the intent, and a
   random response for that intent is shown. If nothing is confident enough, it falls back to
   "Sorry, can't understand you".

## Setup

Requires Python 3.10–3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','wordnet','omw-1.4']]"
```

> **macOS:** if the NLTK download fails with `CERTIFICATE_VERIFY_FAILED`, run
> `pip install certifi && export SSL_CERT_FILE=$(python -m certifi)` and try again.

## Run

```bash
python train_chatbot.py   # re-run whenever you edit intents.json
python chatgui.py
```

## Tech stack

Python · NLTK · TensorFlow/Keras · NumPy · Tkinter

## Limitations

This is a learning project: replies are fixed text, the "modules" it mentions don't exist,
it doesn't remember earlier messages, and it only understands words similar to its examples.
