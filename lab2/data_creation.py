import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


pokemons_dataset_name = "unexpectedscepticism/11945-pokemon-from-first-gen"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


def reset_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_pokemons(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    return kagglehub.dataset_download(pokemons_dataset_name, output_dir=str(save_dir))


def collect(source_dir: Path):
    files = []
    labels = []

    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            for img in class_dir.iterdir():
                files.append(img)
                labels.append(class_dir.name)
    return files, labels




def copy_files(file_list, target_root):
    for file in file_list:
        class_name = file.parent.name
        target_dir = target_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, target_dir / file.name)


def prepare_pokemons(
    save_dir: Path = RAW_DIR,
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
):
    reset_directory(train_dir)
    reset_directory(test_dir)
    download_pokemons(save_dir)
    source_dir = save_dir / "PokemonData"
    files, labels = collect(source_dir)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42, stratify=labels)
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)

if __name__ == "__main__":
    prepare_pokemons()