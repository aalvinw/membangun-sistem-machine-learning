import pandas as pd

# 1. Load data mentah (ubah nama file sesuai milikmu jika perlu)
df = pd.read_csv("Train.csv")

# 2. Lihat kolom dan data unik untuk eksplorasi awal
print("Kolom awal:", df.columns.tolist())
print("Jumlah data:", len(df))

# 3. Hapus kolom yang tidak relevan (contoh: ID pengiriman)
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)

# 4. Hapus baris yang memiliki nilai kosong (missing values)
df.dropna(inplace=True)

# 5. Pastikan label target tersedia
if 'Reached.on.Time_Y.N' not in df.columns:
    raise ValueError("Kolom 'Reached.on.Time_Y.N' tidak ditemukan dalam data!")

# 6. Konversi kolom kategorikal menjadi numerik (one-hot encoding)
categorical_columns = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_columns)

# 7. Simpan data yang sudah diproses ke file baru
df.to_csv("preprocessed_data.csv", index=False)
print("✅ Preprocessing selesai! File disimpan sebagai 'preprocessed_data.csv'.")
