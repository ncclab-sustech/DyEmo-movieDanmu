import pickle
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_sequence

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# postprocess_test.pkl used here can be downloaded from DyEmo dataset on the ScienceDB platform
# https://www.scidb.cn/en/detail?dataSetId=077be74c4cb44fd9bad96ddb4d1520df&version=V1

# ---------------------------------------------------------
# 0. SET RANDOM SEEDS FOR REPRODUCIBILITY
# ---------------------------------------------------------
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed before any random operations
# 89
RANDOM_SEED = 89
set_seed(RANDOM_SEED)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
with open('postprocess_test.pkl', 'rb') as f:
    loaded_variables = pickle.load(f)

scores_allvideo = loaded_variables['scores_allvideo']
smoothed = loaded_variables['smoothed']
all_video_info = loaded_variables['all_video_info']

with open('labels.pkl', 'rb') as f:
    loaded_variables = pickle.load(f)

pe_dict = loaded_variables['pe_dict']
ar_dict = loaded_variables['ar_dict']
va_dict = loaded_variables['va_dict']


# ---------------------------------------------------------
# 2. BUILD DATA FOR ONLY 'pe'
# ---------------------------------------------------------
X_seq = []
y_pe = []

for clip_name, sample in smoothed.items():
    X_seq.append(torch.tensor(sample, dtype=torch.float32))
    y_pe.append(pe_dict[clip_name] - 1)   # shift to 0–12 index

y_pe = np.array(y_pe)


# ---------------------------------------------------------
# 3. DATASET / DATALOADER
# ---------------------------------------------------------
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn_eval(batch):
    """No-noise collate for evaluation."""
    sequences, labels = zip(*batch)
    packed = pack_sequence(sequences, enforce_sorted=False)
    return packed, torch.tensor(labels, dtype=torch.long)

def make_collate_fn_train(noise_std=0.05):
    """Return a collate_fn that adds Gaussian noise to X."""
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        noisy_sequences = []
        for seq in sequences:
            # seq: (T, D) tensor on CPU
            noise = torch.randn_like(seq) * noise_std
            noisy_sequences.append(seq + noise)
        packed = pack_sequence(noisy_sequences, enforce_sorted=False)
        return packed, torch.tensor(labels, dtype=torch.long)
    return collate_fn

dataset = SeqDataset(X_seq, y_pe)

# Split train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
split_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=split_generator
)


# # ---------------------------------------------------------
# # 4. SAMPLER (Balanced Sampler)
# # ---------------------------------------------------------
def effective_num_weights(class_counts, beta=0.999):
    cn = np.array(class_counts)
    effective_num = 1.0 - np.power(beta, cn)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(cn)
    return torch.tensor(weights, dtype=torch.float32)

num_classes = 13
class_counts = np.bincount(y_pe, minlength=num_classes)


sampling_weights = effective_num_weights(class_counts, beta=0.8)
sample_weights = sampling_weights[y_pe]


train_indices = train_dataset.indices
train_sample_weights = torch.tensor(sample_weights[train_indices], dtype=torch.float32)

# Create a generator for the sampler to ensure reproducibility
sampler_generator = torch.Generator().manual_seed(RANDOM_SEED)
sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True,
    generator=sampler_generator
)


# ---------------------------------------------------------
# 5. DATALOADERS
# ---------------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler,
    # collate_fn=collate_fn_eval
    collate_fn=make_collate_fn_train(noise_std=0.87),  # ⬅ noise here
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn_eval,                        # ⬅ no noise
)


# ---------------------------------------------------------
# 6. MODEL
# ---------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, packed_sequences):
        _, hidden = self.gru(packed_sequences)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)


input_size = X_seq[0].size(1)
hidden_size = 256
model = GRUModel(input_size, hidden_size, num_classes).to(device)


# ---------------------------------------------------------
# 7. EFFECTIVE NUMBER CLASS WEIGHTS
# --------------------------------------------------------


class_weights = effective_num_weights(class_counts,beta=0.9992).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.00008)


# ---------------------------------------------------------
# 8. TRAINING
# ---------------------------------------------------------
num_epochs = 25
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for packed_sequences, y in train_loader:
        y = y.to(device)
        outputs = model(packed_sequences.to(device))
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")


# ---------------------------------------------------------
# 9. EVALUATION
# ---------------------------------------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for packed_sequences, y in test_loader:
        y = y.to(device)
        outputs = model(packed_sequences.to(device))
        _, preds = torch.max(outputs, 1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

print("\nFinal Results (PE):")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")


# ---------------------------------------------------------
# 10. LOSS CURVE
# ---------------------------------------------------------
# plt.plot(range(1, num_epochs+1), train_losses, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss (PE)')
# plt.grid(True)
# plt.show()
