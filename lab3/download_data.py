"""
Script to download the UCI Heart Disease Dataset from Kaggle or alternative sources.
"""
import os
import urllib.request
import pandas as pd

OUTPUT_DIR = "data"


def download_from_uci():
    """Download the Cleveland heart disease dataset directly from UCI repository."""
    print("Попытка загрузки из UCI Repository...")
    
    # UCI Machine Learning Repository - Heart Disease Dataset
    url = "https://archive.ics.uci.edu/static/public/45/heart+disease.csv"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "heart_disease_cleveland.csv")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"   ✓ Загружено: {output_path}")
        
        # Verify the file
        df = pd.read_csv(output_path)
        print(f"   Размер: {df.shape}")
        return True
    except Exception as e:
        print(f"   ✗ Не удалось загрузить из UCI: {e}")
        return False


def download_from_kaggle_api():
    """Download using Kaggle API."""
    print("Попытка загрузки через Kaggle API...")
    
    try:
        import kaggleapi
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        api.dataset_download_files(
            'hamawaw112222/uci-heart-disease-dataset',
            path=OUTPUT_DIR,
            unzip=True
        )
        print(f"   ✓ Загружено из Kaggle")
        return True
    except ImportError:
        print("   ✗ kaggle API не установлен")
        return False
    except Exception as e:
        print(f"   ✗ Ошибка Kaggle API: {e}")
        return False


def create_synthetic_dataset():
    """Create a synthetic dataset matching the UCI Heart Disease Dataset schema."""
    print("Создание синтетического датасета (для тестирования)...")
    
    import numpy as np
    
    np.random.seed(42)
    n_samples = 303
    
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.15, 0.50, 0.10, 0.10, 0.15]),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 565, n_samples),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.15, 0.55, 0.30]),
        'thalach': np.random.randint(71, 202, n_samples),
        'exang': np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.25, 0.20]),
        'ca': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.48, 0.30, 0.13, 0.06, 0.03]),
        'thal': np.random.choice([1, 2, 3, 6, 7], n_samples, p=[0.15, 0.55, 0.15, 0.10, 0.05]),
    }
    
    # Generate target based on feature correlations (simplified)
    target = (
        (data['age'] > 50).astype(int) +
        (data['cp'] > 1).astype(int) +
        (data['thalach'] < 140).astype(int) +
        (data['oldpeak'] > 2).astype(int) +
        (data['ca'] > 1).astype(int) +
        np.random.choice([-1, 0, 1], n_samples) * 0.3
    )
    target = (target >= 2).astype(int)
    
    data['target'] = target
    
    df = pd.DataFrame(data)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "heart_disease_cleveland.csv")
    df.to_csv(output_path, index=False)
    
    print(f"   Создан: {output_path}")
    print(f"   Размер: {df.shape}")
    print(f"   Баланс классов: {df['target'].value_counts().to_dict()}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Загрузка UCI Heart Disease Dataset")
    print("=" * 60)
    
    # Try methods in order
    if not download_from_uci():
        print("\nНе удалось загрузить из внешних источников.")
        print("Создаём синтетический датасет для тестирования...\n")
        create_synthetic_dataset()
    
    # List downloaded files
    print(f"\nФайлы в {OUTPUT_DIR}/:")
    for f in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(filepath) / 1024
        print(f"  {f} ({size_mb:.1f} KB)")