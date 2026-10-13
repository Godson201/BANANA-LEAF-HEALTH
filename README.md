# Banana Leaf Disease Classification — FA2

Student: Godson IT  
Registration: BT000601IT26FN

## HTML submission

Open `Godson_IT_BT000601IT26FN_FA2.html` in a browser. It includes all 56 executed code cells and 14 embedded charts/images. Python is not required to view this static report. `HTML_Submission_For_Lecturer.zip` contains this report and the separate `Expanded_Disease_Research.html` report. The notebooks remain the reproducible training and testing source.

## Working dashboard

The frontend is HTML, CSS and JavaScript; the Flask backend loads the saved model and analyzes new photos locally. Start the backend by double-clicking `Start_Dashboard.cmd`, then open http://127.0.0.1:5050. Keep the terminal open. Open the dashboard through that address rather than double-clicking its HTML file.

On a new computer, use Python 3.12 and install `requirements-banana.txt` and `dashboard/requirements.txt` in a `.venv` environment. If the subject-screening weights are not included, run `python tools/prepare_leaf_screening.py` once from that environment; it downloads about 600 MB. Subsequent model inference runs offline. No uploaded photos are saved by the application. Nothing is deployed.

First-time setup from the extracted project folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-banana.txt -r dashboard/requirements.txt
.venv\Scripts\python dashboard/app.py
```

Then open http://127.0.0.1:5050. Internet is needed for dependency installation; the complete project ZIP includes the model weights for offline inference. Keep the server terminal open while using the dashboard.

### Running a GitHub checkout

The repository includes the two small disease-classification checkpoints, example photos, evaluation files and source code. It excludes the local `.venv`, full datasets, sharing ZIPs and the approximately 600 MB third-party subject-screening model. After installing dependencies, prepare that model once:

```powershell
.venv\Scripts\python tools/prepare_leaf_screening.py
```

Then start `Start_Dashboard.cmd`. Sample buttons use the included example photos if the full dataset is absent. Viewing the saved notebook and HTML results does not require downloading datasets. To retrain, obtain the original datasets cited below and restore the paths in the saved split manifests; those datasets are not committed to GitHub.

## Prediction behavior

The default model recognizes four classes. The optional expanded research model recognizes six, adding Fusarium/Panama and Moko bacterial wilt. Low scores or crop disagreement mark the result tentative. Unrecognized subjects show Uncertain without disease scores. If a genuine banana leaf is missed, explicit confirmation assigns a category and records that the user confirmed plant identity. This override must not be used for another plant. Automatic species recognition is heuristic and can make mistakes.

The dashboard includes an eight-condition health guide, source-linked farmer next steps, a whole-plant symptom checklist, optional field notes, downloadable results and printing. The checklist is rule-based guidance, not an additional AI diagnosis. It recommends extension review for concerning whole-plant symptoms even if the photo model predicts Healthy. Field notes are included in downloads; they are not sent with the checklist or stored by the app.

## Expanded research model

The original assignment model and notebook remain intact. `Expanded_Disease_Research.ipynb` reviews the separately trained six-class model, with eight executed code cells and real outputs. Training and evaluation scripts are in `tools/`.

The research model achieved 98.01% (148/151) held-out accuracy and macro F1 0.9663. Its two new-class test sets contain only 7 Fusarium/Panama and 9 Moko images. The data mix different sources and have no independent farm evaluation. Do not interpret this score as field reliability. Moko is not Xanthomonas wilt; no Fusarium race or TR4 identification is provided. Bunchy top and Xanthomonas wilt remain guidance-only until suitable labeled data and evaluation are available.

Source: Mafi, Md Mafiul Hasan Matin; Sifat, R.M.; Moazzam, Md. Golam Moazzam; Uddin, Mohammad Shorif (2023), [Banana Disease Recognition Dataset](https://doi.org/10.17632/79w2n6b4kf.1), CC BY 4.0, obtained through the [Kaggle mirror](https://www.kaggle.com/datasets/sujaykapadnis/banana-disease-recognition-dataset). Only 41 Panama and 55 Moko original images were used; publisher-augmented copies were excluded. Image files were converted to PNG and labels normalized. Exact image content and perceptual similarity checks were applied before splitting the new classes. The saved split manifest is reused when retraining from the included images.

The available bunchy-top candidate contained only 12 mixed images, including an aphid photograph. The [Zenodo bunchy-top dataset](https://zenodo.org/records/7860962) requires access approval. These sources were not used to train a bunchy-top class. See `MODEL_EXPANSION.md` for coverage and limitations.

## Evaluation and sources

The original disease model achieved 97.78% accuracy (132/135), test loss 0.0377 and macro F1 0.9643 on its banana test set. This is not the accuracy of subject screening or arbitrary user uploads. There are 897 unique dataset images split into 627 training, 135 validation and 135 test images, without image-hash overlap.

Dataset: https://www.kaggle.com/datasets/ntambaraetienne/banana-leaf-diseases-classification-dataset  
Classifier: https://docs.pytorch.org/vision/stable/models/mobilenetv3.html  
Subject screen: https://huggingface.co/openai/clip-vit-base-patch32

Retain source attribution and acknowledge assistance according to institutional requirements.
