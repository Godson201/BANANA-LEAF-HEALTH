"""Train a six-class research preview without overwriting the assignment model.

Only original images are used. The source provides Moko (not Xanthomonas wilt)
and Panama disease (not a confirmed Fusarium race). Bunchy top is not trained.
"""
from pathlib import Path
from io import BytesIO
import zipfile
import hashlib
import csv
import json
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'output/expanded_preview'
out.mkdir(exist_ok=True)
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
classes = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka', 'fusarium_wilt', 'moko_wilt']
checkpoint = torch.load(root / 'output/banana_leaf_disease_model.pt', map_location='cpu', weights_only=True)
with (root / 'output/split_manifest.csv').open(newline='') as file:
    records = list(csv.DictReader(file))
rows = [{'image': 'data/OriginalSet/' + row['image_id'], 'class_name': row['class_name'], 'split': row['split']} for row in records]
seen = set()
def image_hash(photo):
    return hashlib.sha256(photo.convert('RGB').resize((128, 128)).tobytes()).hexdigest()
def difference_hash(photo):
    pixels = np.asarray(photo.convert('L').resize((9, 8)), dtype=np.int16)
    return int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), 'big')
for row in rows:
    with Image.open(root / row['image']) as photo:
        seen.add(image_hash(photo))
audit = {'source': 'https://data.mendeley.com/datasets/79w2n6b4kf/1', 'license': 'CC BY 4.0',
         'policy': 'Original images only; no publisher-provided augmented images.', 'excluded_duplicates': 0, 'classes': {}}
archive = root / 'data/expansion_candidates/sujaykapadnis.zip'
if archive.exists():
    with zipfile.ZipFile(archive) as bundle:
        for folder_name, label in [('Banana Panama Disease', 'fusarium_wilt'), ('Banana Moko Disease', 'moko_wilt')]:
            candidates = []
            for entry in bundle.infolist():
                if '/Original Images/' not in entry.filename or Path(entry.filename).parent.name != folder_name:
                    continue
                with Image.open(BytesIO(bundle.read(entry))) as image:
                    photo = image.convert('RGB')
                    digest = image_hash(photo)
                    if digest in seen:
                        audit['excluded_duplicates'] += 1
                        continue
                    seen.add(digest)
                    target = root / 'data/expanded_originals' / label / (digest[:16] + '.png')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    photo.save(target)
                    candidates.append({'image': target.relative_to(root).as_posix(), 'class_name': label, 'dhash': difference_hash(photo)})
            # Keep similar views in the same split, not on both sides of evaluation.
            groups = []
            for candidate in candidates:
                matching = [group for group in groups if any((candidate['dhash'] ^ row['dhash']).bit_count() <= 4 for row in group)]
                if matching:
                    merged = [candidate]
                    for group in matching:
                        merged.extend(group); groups.remove(group)
                    groups.append(merged)
                else:
                    groups.append([candidate])
            assert len(groups) >= 15, f'Too few independent image groups for {label}'
            train, remainder = train_test_split(list(range(len(groups))), test_size=.30, random_state=42)
            val, test = train_test_split(remainder, test_size=.50, random_state=42)
            for split, indices in [('training', train), ('validation', val), ('test', test)]:
                for index in indices:
                    rows.extend({key: value for key, value in row.items() if key != 'dhash'} | {'split': split} for row in groups[index])
            audit['classes'][label] = {'original_images': len(candidates), 'similarity_groups': len(groups)}
else:
    # The portable project contains the selected originals and their fixed split.
    with (out / 'split_manifest.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    audit = json.loads((out / 'data_audit.json').read_text())
    assert len(rows) == 993 and all((root / row['image']).is_file() for row in rows)
with (out / 'split_manifest.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['image', 'class_name', 'split']); writer.writeheader(); writer.writerows(rows)
(out / 'data_audit.json').write_text(json.dumps(audit, indent=2))
print('Data audit:', audit, flush=True)

model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)
model.load_state_dict(checkpoint['model_state'])
old_output = model.classifier[-1]
model.classifier[-1] = nn.Linear(old_output.in_features, len(classes))
with torch.no_grad():
    model.classifier[-1].weight[:4].copy_(old_output.weight)
    model.classifier[-1].bias[:4].copy_(old_output.bias)
model.eval()
for parameter in model.features.parameters():
    parameter.requires_grad = False
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(checkpoint['mean'], checkpoint['std'])])
base = [transforms.Resize(256), transforms.CenterCrop(224)]
training_transform = transforms.Compose(base + [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(12), normalize])
evaluation_transform = transforms.Compose(base + [normalize])
class Photos(Dataset):
    def __init__(self, selected, transform, repeats=1):
        self.rows = selected * repeats; self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(root / row['image']) as photo:
            return self.transform(photo.convert('RGB')), classes.index(row['class_name'])
features = {}
for split in ['training', 'validation', 'test']:
    selected = [row for row in rows if row['split'] == split]
    loader = DataLoader(Photos(selected, training_transform if split == 'training' else evaluation_transform, 3 if split == 'training' else 1), batch_size=32)
    vectors, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            encoded = torch.flatten(model.avgpool(model.features(images)), 1)
            vectors.append(encoded); labels.append(targets)
    features[split] = (torch.cat(vectors), torch.cat(labels))
    print('Encoded', split, len(selected), 'originals', flush=True)

counts = torch.bincount(features['training'][1], minlength=len(classes)).float()
criterion = nn.CrossEntropyLoss(weight=counts.sum() / (len(classes) * counts))
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=.0003, weight_decay=.0001)
loader = DataLoader(TensorDataset(*features['training']), batch_size=64, shuffle=True)
history = []; best_loss = float('inf'); best_state = None; patience = 0
for epoch in range(1, 31):
    model.classifier.train(); total_loss = 0.
    for vectors, targets in loader:
        optimizer.zero_grad(); loss = criterion(model.classifier(vectors), targets); loss.backward(); optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
    model.classifier.eval()
    with torch.inference_mode():
        logits = model.classifier(features['validation'][0])
        val_loss = float(criterion(logits, features['validation'][1]))
        val_accuracy = float((logits.argmax(1) == features['validation'][1]).float().mean())
    history.append({'epoch': epoch, 'training_loss': total_loss / len(features['training'][1]), 'validation_loss': val_loss, 'validation_accuracy': val_accuracy})
    print('Epoch', epoch, 'validation loss', round(val_loss, 4), 'accuracy', round(val_accuracy, 4), flush=True)
    if val_loss < best_loss - .0001:
        best_loss = val_loss; best_state = copy.deepcopy(model.classifier.state_dict()); best_epoch = epoch; patience = 0
    else:
        patience += 1
        if patience >= 6: break
model.classifier.load_state_dict(best_state); model.eval()
with torch.inference_mode():
    test_logits = model.classifier(features['test'][0])
    probabilities = test_logits.softmax(1).numpy()
true = features['test'][1].numpy(); predicted = probabilities.argmax(1)
metrics = {'status': 'experimental', 'class_names': classes, 'test_accuracy': accuracy_score(true, predicted),
           'macro_f1': f1_score(true, predicted, average='macro'), 'test_images': len(true), 'best_epoch': best_epoch,
           'test_loss': float(nn.functional.cross_entropy(test_logits, features['test'][1])), 'actual_epochs': len(history),
           'unique_images': len(rows), 'training_images': sum(row['split']=='training' for row in rows),
           'validation_images': sum(row['split']=='validation' for row in rows),
           'limitations': 'Small new classes from a different source; no independent farm evaluation. Moko is not Xanthomonas wilt. Fusarium race is not identified. Bunchy top is not trained.',
           'per_class': classification_report(true, predicted, target_names=classes, output_dict=True, zero_division=0)}
checkpoint.update(class_names=classes, model_state=model.state_dict(), test_accuracy=metrics['test_accuracy'], best_epoch=best_epoch,
                  test_loss=metrics['test_loss'], actual_epochs=metrics['actual_epochs'], model_status='experimental', limitations=metrics['limitations'])
torch.save(checkpoint, out / 'banana_leaf_expanded_model.pt')
(out / 'evaluation_metrics.json').write_text(json.dumps(metrics, indent=2))
with (out / 'training_history.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
fig, ax = plt.subplots(figsize=(8, 6))
matrix = confusion_matrix(true, predicted, labels=list(range(len(classes))))
ax.imshow(matrix, cmap='Greens'); ax.set_xticks(range(len(classes)), classes, rotation=40, ha='right'); ax.set_yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)): ax.text(j, i, str(matrix[i,j]), ha='center', va='center', color='black')
ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title('Experimental expanded model — held-out images')
fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=150); plt.close(fig)
print('Experimental model saved. Test accuracy:', metrics['test_accuracy'], 'macro F1:', metrics['macro_f1'], flush=True)
