import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import json

# ---------- Архитектура (та же) ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ---------- Классы (можно загрузить из JSON) ----------
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

# ---------- Тестер ----------
class PokemonTester:
    def __init__(self, model_path, class_names, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = SimpleCNN(num_classes=self.num_classes)
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

    def predict_image(self, image_path, top_k=1):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, top_k)
        results = [(self.class_names[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
        return results

    def predict_folder(self, folder_path, top_k=1):
        folder = Path(folder_path)
        image_paths = [p for p in folder.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        for img_path in image_paths:
            try:
                preds = self.predict_image(img_path, top_k)
                print(f"{img_path.name}: {preds[0][0]} ({preds[0][1]:.4f})")
            except Exception as e:
                print(f"Ошибка {img_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pth')
    parser.add_argument('--input', required=True)
    parser.add_argument('--classes', help='JSON файл со списком классов')
    parser.add_argument('--top_k', type=int, default=1)
    args = parser.parse_args()

    class_names = get_class_names(args.classes)
    print(f"Загружено {len(class_names)} классов")
    tester = PokemonTester(args.model, class_names)

    input_path = Path(args.input)
    if input_path.is_dir():
        tester.predict_folder(input_path, args.top_k)
    elif input_path.is_file():
        preds = tester.predict_image(input_path, args.top_k)
        for cls, prob in preds:
            print(f"{cls}: {prob:.4f}")
    else:
        print("Путь не существует")