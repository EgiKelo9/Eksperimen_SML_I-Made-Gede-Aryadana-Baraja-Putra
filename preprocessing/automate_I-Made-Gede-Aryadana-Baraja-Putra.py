import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler

DATASET_DIR = "../pinhomedataset_raw/"
OUTPUT_FILE = "pinhomedataset_preprocessing.csv"
ARTIFACTS_DIR = "artifacts/"

def load_dataset(base_dir="../pinhomedataset_raw/"):
    print("1. Memuat dataset...")
    BASE_DIR = base_dir
    dfs = []
    
    if not os.path.exists(BASE_DIR):
        print(f"Direktori {BASE_DIR} tidak ditemukan. Membuat contoh data untuk pengujian...")
        os.makedirs(BASE_DIR, exist_ok=True)
        return pd.DataFrame()

    for filename in os.listdir(BASE_DIR):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(BASE_DIR, filename)
            try:
                temp_df = pd.read_json(filepath, lines=True)
                if not temp_df.empty:
                    dfs.append(temp_df)
            except Exception as e:
                print(f"Melewati {filename} - Error: {e}")

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"   Berhasil menggabungkan {len(dfs)} file JSONL. Total data: {len(df)} baris.")
    else:
        df = pd.DataFrame()
        print("   Tidak ada data yang berhasil dimuat.")
    
    return df

def drop_irrelevant_columns(df):
    print("2. Menghapus kolom yang tidak relevan...")
    columns_to_drop = ['source', 'url', 'title', 'address', 'scraped_at', 'hash_id']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def handle_duplicates(df):
    print("3. Menghapus data duplikat...")
    df = df.drop_duplicates()
    return df

def handle_missing_values(df):
    print("4. Menangani missing values...")
    cols_to_check = ['district', 'city', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2']
    
    cols_exist = [c for c in cols_to_check if c in df.columns]
    
    if cols_exist:
        df = df.dropna(subset=cols_exist)
    
    categorical_features = ['district', 'city', 'certificate', 'furnishing']
    cat_exist = [c for c in categorical_features if c in df.columns]
    
    if cat_exist:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_exist] = imputer_cat.fit_transform(df[cat_exist])
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(imputer_cat, f'{ARTIFACTS_DIR}imputer_cat.joblib')
    
    cols_to_drop = [c for c in ['electricity', 'carports'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df

def encode_categorical_variables(df):
    print("5. Melakukan encoding variabel kategorikal...")
    encoded_df = df.copy()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    if 'city' in encoded_df.columns:
        city_encoder = LabelEncoder()
        encoded_df['city_encoded'] = city_encoder.fit_transform(encoded_df['city'])
        joblib.dump(city_encoder, f'{ARTIFACTS_DIR}city_encoder.joblib')
    
    if 'district' in encoded_df.columns:
        district_encoder = LabelEncoder()
        encoded_df['district_encoded'] = district_encoder.fit_transform(encoded_df['district'])
        joblib.dump(district_encoder, f'{ARTIFACTS_DIR}district_encoder.joblib')
    
    if 'certificate' in encoded_df.columns:
        certificate_encoder = LabelEncoder()
        encoded_df['certificate_encoded'] = certificate_encoder.fit_transform(encoded_df['certificate'])
        joblib.dump(certificate_encoder, f'{ARTIFACTS_DIR}certificate_encoder.joblib')
    
    if 'furnishing' in encoded_df.columns:
        furnishing_ordinal = ['Unfurnished', 'Semi Furnished', 'Furnished']
        encoded_df['furnishing'] = encoded_df['furnishing'].fillna('Unfurnished') 
        
        unique_furnishings = encoded_df['furnishing'].unique()
        for uf in unique_furnishings:
            if uf not in furnishing_ordinal:
                encoded_df['furnishing'] = encoded_df['furnishing'].replace(uf, 'Unfurnished')

        furnishing_encoder = OrdinalEncoder(categories=[furnishing_ordinal], dtype=int)
        encoded_df['furnishing_encoded'] = furnishing_encoder.fit_transform(encoded_df[['furnishing']])
        joblib.dump(furnishing_encoder, f'{ARTIFACTS_DIR}furnishing_encoder.joblib')
    
    return encoded_df

def handle_outliers(df):
    print("6. Menangani outliers...")
    if 'price_in_rp' in df.columns and 'bedrooms' in df.columns and 'bathrooms' in df.columns and 'building_size_m2' in df.columns and 'land_size_m2' in df.columns:
        filtered_df = df[
            (df['price_in_rp'] >= 3e8) & (df['price_in_rp'] <= 5e10) & 
            (df['bedrooms'] <= 20) & (df['bathrooms'] <= 20) &
            (df['building_size_m2'] <= 2000) & (df['land_size_m2'] <= 2000)
        ]
    else:
        filtered_df = df.copy()
    
    columns_to_check = ['price_in_rp', 'land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms']
    cols_exist = [c for c in columns_to_check if c in filtered_df.columns]
    
    for column in cols_exist:
        Q1 = filtered_df[column].quantile(0.25)
        Q3 = filtered_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        
        outlier_mask = (filtered_df[column] < lower_bound) | (filtered_df[column] > upper_bound)
        outliers_count = outlier_mask.sum()
        print(f"   Kolom {column}: {outliers_count} outliers terdeteksi & dihapus.")
        filtered_df = filtered_df[~outlier_mask]
        
    return filtered_df

def normalize_numerical_features(df):
    print("7. Melakukan normalisasi data (kecuali price_in_rp)...")
    scaler = RobustScaler()
    numerical_features = ['land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms']
    num_exist = [c for c in numerical_features if c in df.columns]
    
    if num_exist:
        df[num_exist] = scaler.fit_transform(df[num_exist])
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(scaler, f'{ARTIFACTS_DIR}robust_scaler.joblib')
    
    return df

def main():
    print("=== Memulai Pipeline Preprocessing ===")
    
    df = load_dataset(base_dir=DATASET_DIR)
    
    if df.empty:
        print("PERINGATAN: Dataset kosong atau tidak ditemukan. Pipeline berhenti.")
        dummy_df = pd.DataFrame(columns=['dummy'])
        dummy_df.to_csv(OUTPUT_FILE, index=False)
        return
        
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = handle_outliers(df)
    df = encode_categorical_variables(df)
    df = normalize_numerical_features(df)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"=== Pipeline Selesai ===")
    print(f"Dataset final yang telah diproses disimpan sebagai: {OUTPUT_FILE}")
    print(f"Total baris data final: {len(df)}")

if __name__ == "__main__":
    main()