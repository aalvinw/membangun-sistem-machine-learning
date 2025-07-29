# news_preprocessing.py

import pandas as pd
import string
import re
import os
from sklearn.model_selection import train_test_split
import nltk

# Pastikan nltk sudah di-download (hanya sekali perlu dijalankan)
nltk.download('stopwords')
from nltk.corpus import stopwords

# Fungsi cleaning text
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ganda
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def main():
    # Buat folder output
    os.makedirs("preprocessed", exist_ok=True)

    # Load data
    df = pd.read_csv("berita.csv")

    # Hapus data kosong dan duplikat
    df.dropna(subset=['judul', 'isi'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Preprocessing teks
    df['judul_clean'] = df['judul'].apply(clean_text)
    df['isi_clean'] = df['isi'].apply(clean_text)

    # Simpan hasil
    df.to_csv("preprocessed/berita_clean.csv", index=False)

    # Bagi ke train-test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("preprocessed/train.csv", index=False)
    test_df.to_csv("preprocessed/test.csv", index=False)

    print("Preprocessing selesai. File disimpan di folder 'preprocessed'.")

if __name__ == "__main__":
    main()
