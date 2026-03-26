from typing import List, Dict, Set, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class NamesDataset(Dataset):
    def __init__(self, data_dir='names', max_len=20):
        self.countries: List[str] = [
            'Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
            'French', 'German', 'Greek', 'Irish', 'Italian',
            'Japanese', 'Korean', 'Polish', 'Portuguese',
            'Russian', 'Scottish', 'Spanish', 'Vietnamese'
        ]
        self.country_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.countries)}

        self.max_len = max_len
        self.samples: List[Tuple[str, int]] = []

        data_path = Path(data_dir)
        for file_path in data_path.glob('*.txt'):
            country = file_path.stem
            if country not in self.country_to_idx:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.samples.append(
                            (name, self.country_to_idx[country]))
        chars: Set[str] = set()
        for name, _ in self.samples:
            for c in name:
                chars.add(c)
        self.char_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(sorted(chars))}
        self.vocab_size = len(self.char_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, country_idx = self.samples[idx]
        name_tensor: torch.Tensor = torch.zeros(
            self.max_len, len(self.char_to_idx))
        for i, c in enumerate(name[:self.max_len]):
            name_tensor[i, self.char_to_idx[c]] = 1

        country_tensor: torch.Tensor = torch.zeros(18)
        country_tensor[country_idx] = 1

        return name_tensor, country_tensor, name


if __name__ == '__main__':
    dataset = NamesDataset('names')
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch_names_tensor, batch_country_tensor, batch_names = next(
        iter(train_loader))

    for i in range(len(batch_names)):
        name_str = batch_names[i]
        name_oh = batch_names_tensor[i]
        country_oh = batch_country_tensor[i]

        # Find country name from one-hot vector
        country_idx = torch.argmax(country_oh).item()
        country_name = dataset.countries[country_idx]

        print(f"Name: {name_str}")
        print(f"Country: {country_name}")
        print(f"Name One-Hot Shape: {name_oh.shape}")
        print(f"Name One-Hot Tensor:\n{name_oh}")
        print(f"Country One-Hot Vector: {country_oh}")
        print("-" * 50)
