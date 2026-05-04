import pandas as pd
import re
import nltk
import tkinter as tk
from tkinter import messagebox, font as tkfont


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# ================================
# NLTK downloads
# ================================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ================================
# Preprocessing Pipeline
# ================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Cleaning: Remove URLs, hashtags, mentions, non-alpha chars
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 2. Tokenization using NLTK word_tokenize
    tokens = word_tokenize(text)

    # 3. Stop Words Removal using nltk.corpus.stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # 4. Lemmatization using WordNetLemmatizer
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# ================================
# Load Dataset
# ================================
print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

# expected columns: Category, Message
df = df.dropna(subset=["Message"])

df = df.rename(columns={
    "Category": "label",
    "Message": "text"
})

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ================================
# Preprocess text
# ================================
df["cleaned"] = df["text"].apply(preprocess_text)

# ================================
# Feature Extraction: TF-IDF Vectorization
# ================================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned"]).toarray()
y = df["label"]

# ================================
# Train/Test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Models
# ================================
model_dt  = DecisionTreeClassifier()
model_nb  = MultinomialNB()
model_rf  = RandomForestClassifier(n_estimators=100)
model_lr  = LogisticRegression(max_iter=300)
model_svm = SVC()

model_dt.fit(X_train,  y_train)
model_nb.fit(X_train,  y_train)
model_rf.fit(X_train,  y_train)
model_lr.fit(X_train,  y_train)
model_svm.fit(X_train, y_train)

print("Models trained successfully!")

# ================================
# Accuracy
# ================================
acc_dt  = accuracy_score(y_test, model_dt.predict(X_test))
acc_nb  = accuracy_score(y_test, model_nb.predict(X_test))
acc_rf  = accuracy_score(y_test, model_rf.predict(X_test))
acc_lr  = accuracy_score(y_test, model_lr.predict(X_test))
acc_svm = accuracy_score(y_test, model_svm.predict(X_test))

# ================================
# Color Palette & Styling
# ================================
BG          = "#0F1117"       # near-black background
CARD        = "#1A1D27"       # card / panel background
ACCENT      = "#C0392B"       # deep crimson accent
ACCENT2     = "#E74C3C"       # lighter red for hover
TEXT_PRI    = "#EAEAEA"       # primary text
TEXT_SEC    = "#8A8FA8"       # secondary / muted text
BORDER      = "#2C2F3F"       # subtle border
SUCCESS_BG  = "#0D2B1A"       # green-tinted bg for ham result
SUCCESS_FG  = "#2ECC71"
DANGER_BG   = "#2B0D0D"       # red-tinted bg for spam result
DANGER_FG   = "#E74C3C"
NEUTRAL_BG  = "#1A1D27"
NEUTRAL_FG  = "#8A8FA8"

# ================================
# GUI Setup
# ================================
root = tk.Tk()
root.title("Spam Detection System")
root.geometry("820x530")
root.configure(bg=BG)
root.resizable(False, False)

# Fonts
title_font   = tkfont.Font(family="Courier New", size=15, weight="bold")
label_font   = tkfont.Font(family="Courier New", size=9,  weight="bold")
input_font   = tkfont.Font(family="Courier New", size=10)
result_font  = tkfont.Font(family="Courier New", size=10, weight="bold")
sub_font     = tkfont.Font(family="Courier New", size=8)

# ================================
# Header
# ================================
header = tk.Frame(root, bg=ACCENT, height=4)
header.pack(fill="x")

title_frame = tk.Frame(root, bg=BG)
title_frame.pack(fill="x", padx=24, pady=(16, 4))

tk.Label(
    title_frame,
    text="⬛ SPAM DETECTION SYSTEM",
    font=title_font,
    bg=BG, fg=TEXT_PRI
).pack(side="left")

tk.Label(
    title_frame,
    text="TF-IDF · 5 Classifiers",
    font=sub_font,
    bg=BG, fg=TEXT_SEC
).pack(side="right", pady=4)

# Divider
tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=24)

# ================================
# Input Section Label
# ================================
tk.Label(
    root,
    text="INPUT MESSAGE",
    font=label_font,
    bg=BG, fg=TEXT_SEC,
    anchor="w"
).pack(fill="x", padx=24, pady=(14, 4))

# ================================
# Input Row: Text box LEFT, Button RIGHT — same line
# ================================
input_row = tk.Frame(root, bg=BG)
input_row.pack(fill="x", padx=24, pady=(0, 6))

# Text area with border frame
text_border = tk.Frame(input_row, bg=ACCENT, padx=1, pady=1)
text_border.pack(side="left", fill="both", expand=True)

text_inner = tk.Frame(text_border, bg=CARD)
text_inner.pack(fill="both", expand=True)

text_box = tk.Text(
    text_inner,
    height=5,
    font=input_font,
    bg=CARD,
    fg=TEXT_PRI,
    insertbackground=ACCENT,
    relief="flat",
    padx=10,
    pady=8,
    wrap="word",
    bd=0,
    highlightthickness=0
)
text_box.pack(fill="both", expand=True)

# ================================
# Prediction Function
# ================================
def predict():
    text = text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter a message first.")
        return

    cleaned = preprocess_text(text)
    vec     = vectorizer.transform([cleaned]).toarray()

    preds = [
        model_dt.predict(vec)[0],
        model_nb.predict(vec)[0],
        model_rf.predict(vec)[0],
        model_lr.predict(vec)[0],
        model_svm.predict(vec)[0],
    ]
    accs = [acc_dt, acc_nb, acc_rf, acc_lr, acc_svm]

    for i, (entry, badge, pred, acc) in enumerate(zip(result_entries, result_badges, preds, accs)):
        if pred == 1:
            tag     = "SPAM"
            e_bg    = DANGER_BG
            e_fg    = DANGER_FG
            b_bg    = DANGER_BG
            b_fg    = DANGER_FG
        else:
            tag     = "HAM"
            e_bg    = SUCCESS_BG
            e_fg    = SUCCESS_FG
            b_bg    = SUCCESS_BG
            b_fg    = SUCCESS_FG

        badge.config(text=tag, bg=b_bg, fg=b_fg)

        entry.config(state="normal")
        entry.delete(0, tk.END)
        entry.insert(0, f"Accuracy: {acc * 100:.2f}%")
        entry.config(
            bg=e_bg,
            fg=e_fg,
            disabledbackground=e_bg,
            disabledforeground=e_fg
        )
        entry.config(state="disabled")

# ================================
# Process Button (right of text box)
# ================================
btn_frame = tk.Frame(input_row, bg=BG)
btn_frame.pack(side="left", padx=(10, 0))

process_btn = tk.Button(
    btn_frame,
    text="PROCESS",
    command=predict,
    font=label_font,
    bg=ACCENT,
    fg="white",
    activebackground=ACCENT2,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    width=10,
    height=4,
    bd=0,
    highlightthickness=0
)
process_btn.pack()

def on_enter(e): process_btn.config(bg=ACCENT2)
def on_leave(e): process_btn.config(bg=ACCENT)
process_btn.bind("<Enter>", on_enter)
process_btn.bind("<Leave>", on_leave)

# ================================
# Results Section Label
# ================================
tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(8, 0))

tk.Label(
    root,
    text="CLASSIFICATION RESULTS",
    font=label_font,
    bg=BG, fg=TEXT_SEC,
    anchor="w"
).pack(fill="x", padx=24, pady=(12, 6))

# ================================
# Result Rows — each algorithm on its OWN line
# ================================
algorithms = [
    ("Decision Tree",       "DT"),
    ("Naive Bayes",         "NB"),
    ("Random Forest",       "RF"),
    ("Logistic Regression", "LR"),
    ("SVM",                 "SVM"),
]

result_entries = []
result_badges  = []

for name, short in algorithms:
    row = tk.Frame(root, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
    row.pack(fill="x", padx=24, pady=3, ipady=6)

    # Algorithm label
    tk.Label(
        row,
        text=f"  {name}",
        font=label_font,
        bg=CARD, fg=TEXT_PRI,
        width=20,
        anchor="w"
    ).pack(side="left", padx=(4, 0))

    # Badge (SPAM / HAM)
    badge = tk.Label(
        row,
        text="—",
        font=sub_font,
        bg=NEUTRAL_BG, fg=NEUTRAL_FG,
        width=6,
        relief="flat",
        padx=6, pady=2
    )
    badge.pack(side="left", padx=8)

    # Accuracy entry
    entry = tk.Entry(
        row,
        font=result_font,
        width=24,
        bg=NEUTRAL_BG,
        fg=NEUTRAL_FG,
        disabledbackground=NEUTRAL_BG,
        disabledforeground=NEUTRAL_FG,
        relief="flat",
        bd=0,
        highlightthickness=0,
        justify="center"
    )
    entry.insert(0, "Awaiting input...")
    entry.config(state="disabled")
    entry.pack(side="left", padx=4)

    result_entries.append(entry)
    result_badges.append(badge)

# ================================
# Footer
# ================================
tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(10, 0))
tk.Label(
    root,
    text="NLP Pipeline: Cleaning → Tokenization → Stop Words → Lemmatization → TF-IDF",
    font=sub_font,
    bg=BG, fg=TEXT_SEC
).pack(pady=8)

print("GUI Running...")
root.mainloop()