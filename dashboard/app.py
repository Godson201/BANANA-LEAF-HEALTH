"""Local Flask backend for the HTML and CSS banana-leaf dashboard."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import csv
import json
import threading
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms
from leaf_screening import screen_photo, screening_ready, load_screening_model, ScreeningUnavailable
from farmer_guidance import CATALOG, SYMPTOMS, care_for, field_check

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024 + 65536
lock = threading.Lock()
torch.set_num_threads(4)

@lru_cache(maxsize=1)
def load_predictor():
    checkpoint = torch.load(OUTPUT / 'banana_leaf_disease_model.pt', map_location='cpu', weights_only=True)
    classes = checkpoint['class_names']
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(checkpoint['mean'], checkpoint['std'])])
    return model, normalize, checkpoint

@lru_cache(maxsize=1)
def load_expanded_predictor():
    checkpoint = torch.load(OUTPUT / 'expanded_preview/banana_leaf_expanded_model.pt', map_location='cpu', weights_only=True)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state']); model.eval()
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(checkpoint['mean'], checkpoint['std'])])
    return model, normalize, checkpoint

def predict(photo, multiple_crops=False, model_choice='baseline'):
    model, normalize, checkpoint = load_predictor() if model_choice == 'baseline' else load_expanded_predictor()
    resized = transforms.Resize(checkpoint['resize_shorter_side'])(photo.convert('RGB'))
    crops = transforms.FiveCrop(checkpoint['crop_size'])(resized) if multiple_crops else [transforms.CenterCrop(checkpoint['crop_size'])(resized)]
    with lock, torch.inference_mode():
        views = model(torch.stack([normalize(crop) for crop in crops])).softmax(1)
    scores = views.mean(0).tolist()
    best = max(range(len(scores)), key=scores.__getitem__)
    agreement = int((views.argmax(1) == best).sum())
    ordered = sorted(scores, reverse=True)
    tentative = multiple_crops and (agreement < 4 or ordered[0] < .80 or ordered[0] - ordered[1] < .20)
    return {'status': 'accepted', 'predicted_class': checkpoint['class_names'][best], 'confidence': scores[best],
        'model_choice': model_choice, 'experimental': model_choice == 'expanded',
        'probabilities': [{'class_name': name, 'probability': score} for name, score in zip(checkpoint['class_names'], scores)],
        'crop_agreement': {'matching': agreement, 'total': len(crops)}, 'tentative': tentative,
        'message': 'Tentative category: scores are low or crops disagree. Try a clearer photo.' if tentative else 'The banana leaf received a predicted category.'}

def assess_upload(photo, banana_confirmed=False, model_choice='baseline'):
    screening = {'accepted': True, 'user_confirmed': True, 'method': 'Banana identity confirmed by user'} if banana_confirmed else screen_photo(photo)
    if not screening['accepted']:
        return {'status': 'uncertain', 'predicted_class': None, 'confidence': None, 'probabilities': [],
            'message': screening['message'], 'screening': screening, 'can_confirm_banana': screening.get('reason') != 'too_small'}
    result = predict(photo, multiple_crops=True, model_choice=model_choice)
    return {**result, 'screening': screening, 'farmer_guidance': care_for(result['predicted_class'], result['tentative'])}

@app.get('/api/diseases')
def disease_library():
    return jsonify(diseases=CATALOG, predicted_classes=['cordana', 'healthy', 'pestalotiopsis', 'sigatoka'])

@app.get('/api/models')
def model_catalog():
    baseline = json.loads((OUTPUT / 'evaluation_metrics.json').read_text())
    expanded = json.loads((OUTPUT / 'expanded_preview/evaluation_metrics.json').read_text())
    return jsonify(models={
        'baseline': {**baseline, 'class_names': ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka'],
                     'name': 'Leaf-spot model', 'experimental': False,
                     'confusion_chart': 'assets/confusion_matrix.png', 'accuracy_chart': 'assets/accuracy_curve.png',
                     'note': 'Original assignment model. Four trained classes; independent farm evaluation is still needed.'},
        'expanded': {**expanded, 'name': 'Expanded research preview', 'experimental': True,
                     'confusion_chart': 'assets/expanded_confusion_matrix.png', 'accuracy_chart': 'assets/expanded_accuracy_curve.png',
                     'note': 'Experimental: adds Fusarium/Panama and Moko, not Xanthomonas wilt. New-class test samples are small. Bunchy top is not trained.'}})

@app.post('/api/field-check')
def check_field():
    payload = request.get_json(silent=True)
    symptoms = payload.get('symptoms', []) if isinstance(payload, dict) else None
    if not isinstance(symptoms, list) or any(not isinstance(item, str) or item not in SYMPTOMS for item in symptoms):
        return jsonify(error='Choose symptoms from the checklist.'), 400
    return jsonify(field_check(symptoms))

@app.get('/')
def index():
    return send_from_directory(app.root_path, 'index.html')

@app.get('/assets/<path:name>')
def assets(name):
    return send_from_directory(Path(app.root_path) / 'assets', name)

@app.get('/api/health')
def health():
    try:
        _, _, checkpoint = load_predictor()
        return jsonify(status='ready', classes=checkpoint['class_names'], screening_available=screening_ready())
    except (OSError, RuntimeError, KeyError):
        return jsonify(error='Saved classifier unavailable.'), 503

@app.post('/api/predict')
def predict_upload():
    model_choice = request.form.get('model', 'baseline')
    if model_choice not in {'baseline', 'expanded'}:
        return jsonify(error='Choose one of the available models.'), 400
    uploaded = request.files.get('image')
    if uploaded is None or not uploaded.filename:
        return jsonify(error='Choose a leaf photo first.'), 400
    content = uploaded.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        return jsonify(error='Please choose a photo smaller than 8 MB.'), 413
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in {'JPEG', 'PNG', 'WEBP'}:
                return jsonify(error='Choose a JPG, PNG or WebP image.'), 400
            if image.width * image.height > 20_000_000:
                return jsonify(error='Resize this photo to under 20 megapixels.'), 400
            photo = ImageOps.exif_transpose(image).convert('RGB')
            photo.load()
        result = assess_upload(photo, request.form.get('banana_confirmed') == 'true', model_choice)
        return jsonify(**result, image_size={'width': photo.width, 'height': photo.height})
    except ScreeningUnavailable:
        return jsonify(error='Photo screening is not ready. Prepare the screening model before uploading.'), 503
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error='This file could not be read as a supported image.'), 400

@lru_cache(maxsize=1)
def sample_rows():
    with (OUTPUT / 'banana_leaf_predictions.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    return {name: next(row for row in rows if row['true_class'] == name) for name in ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']}

@app.post('/api/sample/<name>')
def sample(name):
    model_choice = request.args.get('model', 'baseline')
    if model_choice not in {'baseline', 'expanded'}:
        return jsonify(error='Choose one of the available models.'), 400
    row = sample_rows().get(name)
    if row is not None:
        path = ROOT / 'data/OriginalSet' / row['image_id']
    elif model_choice == 'expanded' and name in {'fusarium_wilt', 'moko_wilt'}:
        with (OUTPUT / 'expanded_preview/test_predictions.csv').open(newline='') as file:
            row = next((row for row in csv.DictReader(file) if row['true_class'] == name and row['predicted_class'] == name), None)
        if row is None:
            return jsonify(error='Unknown sample.'), 404
        path = ROOT / row['image']
    else:
        return jsonify(error='Unknown sample.'), 404
    if not path.is_file():
        # GitHub includes representative sample photos, not the full datasets.
        path = Path(app.root_path) / 'assets' / (name + '.jpg')
    with Image.open(path) as photo:
        result = predict(photo, model_choice=model_choice)
    return jsonify(**result, true_class=name, screening={'accepted': True, 'method': 'Known banana dataset example'},
                   farmer_guidance=care_for(result['predicted_class'], result['tentative']))

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Please choose a photo smaller than 8 MB.'), 413

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    from waitress import serve
    load_predictor()
    if screening_ready():
        load_screening_model()
    print('LeafLens: http://127.0.0.1:5050', flush=True)
    serve(app, host='127.0.0.1', port=5050, threads=4)
