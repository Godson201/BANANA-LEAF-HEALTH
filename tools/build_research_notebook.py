"""Create and execute a short review notebook for the saved research model."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
markdown = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
notebook.cells = [
markdown('''# Expanded banana disease model — research preview
**Godson IT**

This extension adds the dataset's **Panama/Fusarium** and **Moko bacterial wilt** labels to the original four classes. Moko is not Xanthomonas wilt. Bunchy top is not trained. The new-class samples are small, so this is not a field-validated diagnostic model.

Original-image source: Mafi, Sifat, Moazzam and Uddin (2023), [Banana Disease Recognition Dataset](https://doi.org/10.17632/79w2n6b4kf.1), CC BY 4.0. Publisher-provided augmented copies were excluded. Training code: `tools/train_expanded_preview.py`; reproducible held-out evaluation: `tools/evaluate_expanded_preview.py`. The original assignment model is preserved.
'''),
markdown('## 1. Read the saved experiment files'),
code('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage
folder = Path("output/expanded_preview")
metrics = json.loads((folder / "evaluation_metrics.json").read_text())
audit = json.loads((folder / "data_audit.json").read_text())
print("Reviewing:", folder)
print("Status:", metrics["status"])
'''),
markdown('## 2. Check data provenance and splits\nThe original assignment split is retained. New originals were deduplicated, grouped by image similarity and split before training. Augmentation is applied only to training images.'),
code('''print("Original-image source:", audit["source"])
print("License:", audit["license"])
display(pd.DataFrame(audit["classes"]).T)
manifest = pd.read_csv(folder / "split_manifest.csv")
display(pd.crosstab(manifest.class_name, manifest.split))
assert manifest.image.is_unique
print("Unique image paths:", len(manifest))
'''),
markdown('## 3. Inspect the actual training history\nThe existing feature extractor was frozen. Three augmented training views per image were encoded, and the classifier head was trained with class-weighted cross entropy and Adam. Validation loss selected the checkpoint; early stopping used patience six.'),
code('''history = pd.read_csv(folder / "training_history.csv")
display(history)
print("Selected epoch:", metrics["best_epoch"])
'''),
code('''fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.epoch, history.training_loss, label="Training loss")
ax.plot(history.epoch, history.validation_loss, label="Validation loss")
ax.axvline(metrics["best_epoch"], color="grey", linestyle="--", label="Selected epoch")
ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy loss", title="Research model learning history")
ax.legend(); ax.grid(alpha=.2)
plt.show()
'''),
markdown('## 4. Review held-out results\nThe accuracy below is for this mixed-source dataset. New-class test supports are just seven Fusarium and nine Moko images; no independent farm generalization is established. Evaluation used the saved center-crop preprocessing.'),
code('''print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"Test loss: {metrics['test_loss']:.4f}")
print("Held-out images:", metrics["test_images"])
report = pd.DataFrame({name: metrics["per_class"][name] for name in metrics["class_names"]}).T
display(report)
'''),
code('''display(DisplayImage(filename=str(folder / "confusion_matrix.png")))
print("Rows are true classes; columns are predicted classes.")
'''),
markdown('## 5. Reload the real saved model and test a held-out example'),
code('''import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
torch.set_num_threads(4)
checkpoint = torch.load(folder / "banana_leaf_expanded_model.pt", map_location="cpu", weights_only=True)
model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state"])
model.eval()
preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(checkpoint["mean"], checkpoint["std"])])
print("Saved classes:", checkpoint["class_names"])
'''),
code('''example = manifest[(manifest.split == "test") & (manifest.class_name == "fusarium_wilt")].iloc[0]
with Image.open(example.image) as image:
    photo = image.convert("RGB")
with torch.inference_mode():
    probabilities = model(preprocess(photo).unsqueeze(0)).softmax(1)[0].numpy()
best = int(probabilities.argmax())
print("True dataset label:", example.class_name)
print("Predicted label:", checkpoint["class_names"][best])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(photo); axes[0].axis("off")
axes[1].barh(checkpoint["class_names"], probabilities, color="#2f6c4e")
axes[1].set(xlim=(0,1), xlabel="Uncalibrated model score", title="Six-class research prediction")
fig.tight_layout(); plt.show()
'''),
markdown('''## Interpretation
The dashboard keeps the original four-class model as the default and makes this model an explicit research choice. Neither model confirms a pathogen, Fusarium race, chemical treatment or field diagnosis. Bunchy top and Xanthomonas wilt remain guidance-only. Obtain suitable expert-labeled images and independent farm evaluation before adding those predictions.
''')]
notebook.metadata['kernelspec'] = {'display_name': 'Banana Leaf (PyTorch)', 'language': 'python', 'name': 'banana-leaf'}
client = NotebookClient(notebook, timeout=600, kernel_name='banana-leaf', resources={'metadata': {'path': str(root)}})
client.execute()
nbf.write(notebook, root / 'Expanded_Disease_Research.ipynb')
print('Research notebook executed with', sum(cell.cell_type=='code' for cell in notebook.cells), 'code cells and real outputs.')
