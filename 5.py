import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NamesDataset(Dataset):
    def __init__(self, data_dir='names', max_len=20):
        self.countries: list[str] = [
            'Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
            'French', 'German', 'Greek', 'Irish', 'Italian',
            'Japanese', 'Korean', 'Polish', 'Portuguese',
            'Russian', 'Scottish', 'Spanish', 'Vietnamese'
        ]
        self.country_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.countries)}

        self.samples: list[tuple[str, int]] = []
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
        chars: set[str] = set()
        for name, _ in self.samples:
            for c in name:
                chars.add(c)
        self.char_to_idx: dict[str, int] = {
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

        return name_tensor, torch.tensor(country_idx, dtype=torch.long)


class NameClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          batch_first=True, num_layers=2, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size) when batch_first=True
        # output: output(batch, seq_len, hidden_size*2), h_n(4, batch, hidden_size) when batch_first=True
        _, hidden = self.rnn(x)
        # hidden: (1, batch, hidden_size)
        out = torch.cat((hidden[-2], hidden[-1]),
                        dim=1)  # (batch, hidden_size*2)

        out = self.fc(out)  # (batch, 18)
        return out


def train(model: NameClassifier, train_loader: DataLoader, epochs: int = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
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
    for i, c in enumerate(name[:dataset.max_len]):
        if c in dataset.char_to_idx:
            x[0, i, dataset.char_to_idx[c]] = 1

    with torch.no_grad():
        output = model(x)
        pred_idx = torch.argmax(output, dim=1).item()

    return dataset.countries[int(pred_idx)]


if __name__ == '__main__':
    dataset = NamesDataset('names')
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = NameClassifier(
        input_size=dataset.vocab_size,
        hidden_size=256,
        num_classes=len(dataset.countries)).to(device)
    train(model, train_loader, epochs=30)

    testnames = ["LiuYe", "Liu Ye", "John", "Karev", "Smythson"]
    for name in testnames:
        print(f"{name} -> {predict(model, name, dataset)}")
    # Epoch 1, Loss: 0.0122, Acc: 0.5560
    # Epoch 2, Loss: 0.0091, Acc: 0.6620
    # Epoch 3, Loss: 0.0083, Acc: 0.6877
    # Epoch 4, Loss: 0.0076, Acc: 0.7124
    # Epoch 5, Loss: 0.0069, Acc: 0.7355
    # Epoch 6, Loss: 0.0064, Acc: 0.7526
    # Epoch 7, Loss: 0.0060, Acc: 0.7665
    # Epoch 8, Loss: 0.0057, Acc: 0.7749
    # Epoch 9, Loss: 0.0054, Acc: 0.7870
    # Epoch 10, Loss: 0.0051, Acc: 0.7987
    # Epoch 11, Loss: 0.0048, Acc: 0.8086
    # Epoch 12, Loss: 0.0046, Acc: 0.8161
    # Epoch 13, Loss: 0.0044, Acc: 0.8192
    # Epoch 14, Loss: 0.0043, Acc: 0.8266
    # Epoch 15, Loss: 0.0041, Acc: 0.8324
    # Epoch 16, Loss: 0.0039, Acc: 0.8408
    # Epoch 17, Loss: 0.0037, Acc: 0.8491
    # Epoch 18, Loss: 0.0036, Acc: 0.8532
    # Epoch 19, Loss: 0.0035, Acc: 0.8594
    # Epoch 20, Loss: 0.0033, Acc: 0.8616
    # Epoch 21, Loss: 0.0032, Acc: 0.8689
    # Epoch 22, Loss: 0.0030, Acc: 0.8761
    # Epoch 23, Loss: 0.0029, Acc: 0.8780
    # Epoch 24, Loss: 0.0028, Acc: 0.8839
    # Epoch 25, Loss: 0.0026, Acc: 0.8903
    # Epoch 26, Loss: 0.0026, Acc: 0.8935
    # Epoch 27, Loss: 0.0025, Acc: 0.8990
    # Epoch 28, Loss: 0.0023, Acc: 0.9032
    # Epoch 29, Loss: 0.0023, Acc: 0.9045
    # Epoch 30, Loss: 0.0021, Acc: 0.9122
    # LiuYe -> Chinese
    # Liu Ye -> Chinese
    # John -> English
    # Karev -> Russian
    # Smythson -> English
