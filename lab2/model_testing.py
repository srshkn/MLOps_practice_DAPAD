import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import json
import architecture
import csv

DATASET_DIR = Path(__file__).resolve().parent / 'dataset'
DEFAULT_TEST_DIR = DATASET_DIR / 'test'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'model.pth'


def get_class_names(classes_file=None):
    return sorted([
    'Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno',
    'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree',
    'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon',
    'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett',
    'Ditto', 'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini',
    'Drowzee', 'Dugtrio', 'Eevee', 'Ekans', 'Electabuzz', 'Electrode',
    'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon',
    'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen',
    'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados',
    'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur',
    'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra',
    'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras',
    'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar',
    'Magnemite', 'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod',
    'Mew', 'Mewtwo', 'Moltres', 'Mr.Mime', 'Muk', 'Nidoking', 'Nidoqueen',
    'Nidoran-f', 'Nidoran-m', 'Nidorina', 'Nidorino', 'Ninetales',
    'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect',
    'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir',
    'Poliwag', 'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon',
    'Primeape', 'Psyduck', 'Raichu', 'Rapidash', 'Raticate', 'Rattata',
    'Rhydon', 'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra',
    'Seaking', 'Seel', 'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax',
    'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros',
    'Tentacool', 'Tentacruel', 'Vaporeon', 'Venomoth', 'Venonat',
    'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb', 'Vulpix',
    'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff',
    'Zapdos', 'Zubat'
])

class PokemonTester:
    def __init__(self, model_path, class_names, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = architecture.SimpleCNN(num_classes=self.num_classes)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Модель загружена на {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        """Возвращает предсказанный класс и вероятность."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        top_prob, top_idx = torch.max(probs, dim=0)
        predicted_class = self.class_names[top_idx.item()]
        confidence = top_prob.item()
        return predicted_class, confidence



def evaluate(test_dir, model_path, output_csv, output_json, classes_file=None):
    """Основная функция тестирования."""
    class_names = get_class_names(classes_file)
    print(f"Загружено {len(class_names)} классов")

    tester = PokemonTester(model_path, class_names)

    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Тестовая папка {test_dir} не найдена")

    results = []  # каждый элемент: (image_path, true_class, pred_class, confidence)
    correct = 0
    total = 0

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        true_class = class_dir.name
        if true_class not in class_names:
            print(f"Предупреждение: класс {true_class} не найден в списке, пропускаем")
            continue

        image_paths = list(class_dir.glob('*'))
        for img_path in image_paths:
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                pred_class, confidence = tester.predict_image(img_path)
                results.append({
                    'image': str(img_path),
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': confidence
                })
                total += 1
                if pred_class == true_class:
                    correct += 1
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'true_class', 'pred_class', 'confidence'])
        writer.writeheader()
        writer.writerows(results)

    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Результаты сохранены в {output_csv} и {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Тестирование модели покемонов')
    parser.add_argument('--test_dir', default=str(DEFAULT_TEST_DIR), help='Путь к папке с тестовыми данными (подпапки = классы)')
    parser.add_argument('--model', default=str(DEFAULT_MODEL_PATH), help='Путь к файлу модели')
    parser.add_argument('--output_csv', default='predictions.csv', help='Выходной CSV файл')
    parser.add_argument('--output_json', default='metrics.json', help='Выходной JSON файл')
    parser.add_argument('--classes', help='JSON файл со списком классов (опционально)')
    args = parser.parse_args()

    evaluate(
        test_dir=args.test_dir,
        model_path=args.model,
        output_csv=args.output_csv,
        output_json=args.output_json,
        classes_file=args.classes
    )