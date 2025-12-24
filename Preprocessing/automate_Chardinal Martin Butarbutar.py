from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_transaction_data(
    input_path: str | Path,
    output_path: str | Path,
    verbose: bool = True,
) -> Path:
    """
    Melakukan preprocessing dataset Retail Transaction.

    Tahapan preprocessing:
    1) Menghapus data duplikat
    2) Menghapus baris kosong (Missing Values)
    3) Konversi TransactionDate menjadi tipe DateTime
    4) Rename kolom 'DiscountApplied(%)' -> 'DiscountApplied'
    5) Menghapus kolom yang tidak diperlukan ('StoreLocation' dan 'ProductID')
    6) Encoding kolom kategorikal ('PaymentMethod' dan 'ProductCategory')
    7) Menyimpan hasil preprocessing ke file CSV
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    # Membaca dataset
    if verbose:
        print(f" Membaca dataset dari: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Menghapus Data Duplikat
    if verbose:
        print(f"   Jumlah duplikat sebelum dihapus: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    # 2. Menghapus Baris Kosong (Missing Values)
    if verbose:
        print(f"   Jumlah missing values sebelum dihapus:\n{df.isnull().sum()}")
    df.dropna(inplace=True)

    # 3. Mengubah TransactionDate menjadi tipe DateTime
    # Pastikan kolom ada sebelum diubah
    if 'TransactionDate' in df.columns:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    # 4. Mengubah nama kolom 'DiscountApplied(%)' menjadi 'DiscountApplied'
    df.rename(columns={'DiscountApplied(%)': 'DiscountApplied'}, inplace=True)

    # 5. Menghapus kolom yang tidak diperlukan
    cols_to_drop = ['StoreLocation', 'ProductID']
    # Hanya drop jika kolom tersebut benar-benar ada di dataframe
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)

    # 6. Encoding kolom Kategorikal
    # Catatan: Dalam pipeline produksi, sebaiknya simpan objek LabelEncoder (pickle)
    # agar bisa digunakan untuk inverse_transform atau data baru.
    if 'PaymentMethod' in df.columns:
        le_payment = LabelEncoder()
        df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])

    if 'ProductCategory' in df.columns:
        le_product = LabelEncoder()
        df['ProductCategory'] = le_product.fit_transform(df['ProductCategory'])

    # 7. Menyimpan hasil preprocessing
    # Membuat folder output jika belum ada
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)

    return output_path


def parse_args() -> argparse.Namespace:
    """
    Mengatur argumen command line untuk menjalankan preprocessing otomatis.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing otomatis dataset Retail Transaction"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Retail-Transaction-Dataset_raw.csv",
        help="Path ke dataset mentah (default: Retail-Transaction-Dataset_raw.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/Retail-Transaction-Dataset_clean.csv",
        help="Path untuk menyimpan dataset hasil preprocessing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nonaktifkan log print (verbose)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        output_file = preprocess_transaction_data(
            input_path=args.input,
            output_path=args.output,
            verbose=not args.quiet,
        )
        print(f"✅ Preprocessing selesai. Dataset tersimpan di: {output_file}")
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")