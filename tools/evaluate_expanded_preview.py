"""Save reproducible held-out predictions and plots for the research model."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]
folder = root / 'output/expanded_preview'
torch.set_num_threads(4)
checkpoint = torch.load(folder / 'banana_leaf_expanded_model.pt', map_location='cpu', weights_only=True)
model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(checkpoint['class_names']))
model.load_state_dict(checkpoint['model_state']); model.eval()
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(checkpoint['mean'], checkpoint['std'])])
with (folder / 'split_manifest.csv').open(newline='') as file: rows = list(csv.DictReader(file))
test = [row for row in rows if row['split'] == 'test']
logits = []
with torch.inference_mode():
    for start in range(0, len(test), 32):
        images = []
        for row in test[start:start+32]:
            with Image.open(root / row['image']) as photo: images.append(preprocess(photo.convert('RGB')))
        logits.append(model(torch.stack(images)))
logits = torch.cat(logits)
labels = torch.tensor([checkpoint['class_names'].index(row['class_name']) for row in test])
loss = float(nn.functional.cross_entropy(logits, labels))
probabilities = logits.softmax(1).numpy()
with (folder / 'test_predictions.csv').open('w', newline='') as file:
    writer = csv.writer(file); writer.writerow(['image', 'true_class', 'predicted_class', 'confidence'])
    for row, scores in zip(test, probabilities):
        writer.writerow([row['image'], row['class_name'], checkpoint['class_names'][int(scores.argmax())], float(scores.max())])
metrics = json.loads((folder / 'evaluation_metrics.json').read_text())
with (folder / 'training_history.csv').open(newline='') as file: history = list(csv.DictReader(file))
metrics.update(test_loss=loss, actual_epochs=len(history), unique_images=len(rows),
               training_images=sum(row['split']=='training' for row in rows), validation_images=sum(row['split']=='validation' for row in rows))
checkpoint.update(test_loss=loss, actual_epochs=len(history))
torch.save(checkpoint, folder / 'banana_leaf_expanded_model.pt')
(folder / 'evaluation_metrics.json').write_text(json.dumps(metrics, indent=2))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([int(row['epoch']) for row in history], [float(row['validation_accuracy']) for row in history], marker='o', color='#2f6c4e')
ax.axvline(metrics['best_epoch'], linestyle='--', color='#ad762b', label='Selected epoch')
ax.set(xlabel='Epoch', ylabel='Validation accuracy', title='Experimental model — validation accuracy', ylim=(0,1.02)); ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(folder / 'accuracy_curve.png', dpi=150); plt.close(fig)
print('Research model:', metrics['test_accuracy'], 'test accuracy;', loss, 'test loss;', len(test), 'held-out images')
