from typing import List, Dict, Set, Tuple
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.samples: List[Tuple[str, int]] = []
        self.max_len = max_len

        data_path = Path(data_dir)
        for file_path in data_path.glob('*.txt'):
            country = file_path.stem
            if country not in self.country_to_idx:
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    cleaned_name = []
                    for c in name:
                        if c.isalpha() and c != '\xa0':
                            cleaned_name.append(c)
                    name = ''.join(cleaned_name).strip()
                    if not name:
                        continue
                    self.samples.append(
                        (name, self.country_to_idx[country]))
        chars: Set[str] = set()
        for name, _ in self.samples:
            for c in name:
                chars.add(c)
        self.char_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(sorted(chars))}
        print(self.char_to_idx)
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

        return name_tensor, country_tensor, country_idx, name


class NameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size) when batch_first=True
        _, (hidden, _) = self.rnn(x)
        # hidden: (1, batch, hidden_size)
        out = hidden[-1]  # (batch, hidden_size)

        out = self.fc(out)  # (batch, 18)
        return out


def train(model: NameClassifier, dataset: NamesDataset, train_loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        epoch_loss = 0
        correct = 0
        total = 0
        for x, _, y, _ in train_loader:
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            y_pred = model(x)
            loss: Tensor = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            epoch_loss += loss.item()
        acc = correct / total
        print(
            f"Epoch {epoch+1}, Loss: {epoch_loss / total:.4f}, Acc: {acc:.4f}")


def predict(model: nn.Module, name: str, dataset: NamesDataset):
    model.eval()

    x = torch.zeros(1, dataset.max_len, dataset.vocab_size).to(device)
    print(x.shape)
    for i, c in enumerate(name[:dataset.max_len]):
        if c in dataset.char_to_idx:
            x[0, i, dataset.char_to_idx[c]] = 1

    with torch.no_grad():
        output = model(x)
        pred_idx = torch.argmax(output, dim=1).item()

    return dataset.countries[int(pred_idx)]


if __name__ == '__main__':
    dataset = NamesDataset('names')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = NameClassifier(
        input_size=dataset.vocab_size,
        hidden_size=128,
        num_classes=len(dataset.countries)).to(device)
    train(model, dataset, train_loader)
    print(predict(model, "Liu Ye", dataset))
    # Epoch 1, Loss: 0.0297, Acc: 0.4575
    # Epoch 2, Loss: 0.0289, Acc: 0.4687
    # Epoch 3, Loss: 0.0265, Acc: 0.5012
    # Epoch 4, Loss: 0.0227, Acc: 0.5563
    # Epoch 5, Loss: 0.0198, Acc: 0.6364
    # Epoch 6, Loss: 0.0177, Acc: 0.6708
    # Epoch 7, Loss: 0.0166, Acc: 0.6876
    # Epoch 8, Loss: 0.0155, Acc: 0.7113
    # Epoch 9, Loss: 0.0146, Acc: 0.7283
    # Epoch 10, Loss: 0.0140, Acc: 0.7377
    # torch.Size([1, 20, 79])
    # Chinese
