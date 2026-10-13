"""Check live classifier behavior, subject screening and upload validation."""
from pathlib import Path
import sys
import unittest
import io
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app, ROOT, sample_rows, predict, load_predictor
from leaf_screening import ScreeningUnavailable
from PIL import Image
from sklearn.datasets import load_sample_image
import torch

class DashboardChecks(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def upload(self, photo, confirmed=False):
        content = io.BytesIO()
        photo.save(content, 'PNG')
        return self.client.post('/api/predict', data={'image': (io.BytesIO(content.getvalue()), 'leaf.png'),
                                                    'banana_confirmed': 'true' if confirmed else 'false'})

    def test_health_and_frontend(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertTrue(self.client.get('/api/health').json['screening_available'])
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)

    def test_four_samples_match_saved_evaluation(self):
        for name, row in sample_rows().items():
            result = self.client.post('/api/sample/' + name).json
            self.assertEqual(result['predicted_class'], row['predicted_class'])
            for item in result['probabilities']:
                self.assertAlmostEqual(item['probability'], float(row['probability_' + item['class_name']]), places=4)

    def test_confirmed_banana_receives_category(self):
        with Image.open(ROOT / 'data/OriginalSet/cordana/87.jpeg') as photo:
            result = self.upload(photo.convert('RGB'), confirmed=True).json
        self.assertEqual(result['status'], 'accepted')
        self.assertIn(result['predicted_class'], sample_rows())
        self.assertTrue(result['screening']['user_confirmed'])

    def test_real_other_subjects_are_uncertain(self):
        for name in ['china.jpg', 'flower.jpg']:
            result = self.upload(Image.fromarray(load_sample_image(name))).json
            self.assertEqual(result['status'], 'uncertain')
            self.assertIsNone(result['predicted_class'])
            self.assertEqual(result['probabilities'], [])

    def test_recognized_banana_upload(self):
        with Image.open(ROOT / 'data/OriginalSet/healthy/102.jpeg') as photo:
            response = self.upload(photo.convert('RGB'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['predicted_class'], 'healthy')

    def test_unavailable_screen_does_not_predict(self):
        with patch('app.screen_photo', side_effect=ScreeningUnavailable('Missing')):
            response = self.upload(Image.new('RGB', (256, 256)))
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('predicted_class', response.json)

    def test_crop_disagreement_retains_tentative_category(self):
        _, normalize, checkpoint = load_predictor()
        logits = torch.tensor([[0., 0., 0., 12.]] * 3 + [[0., 12., 0., 0.]] * 2)
        with patch('app.load_predictor', return_value=(lambda batch: logits, normalize, checkpoint)):
            result = predict(Image.new('RGB', (300, 300)), multiple_crops=True)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['predicted_class'], 'sigatoka')
        self.assertTrue(result['tentative'])

    def test_bad_missing_and_oversized_uploads(self):
        self.assertEqual(self.client.post('/api/predict').status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data={'image': (io.BytesIO(b'not an image'), 'fake.jpg')}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data={'image': (io.BytesIO(b'x' * (8 * 1024 * 1024 + 1)), 'large.jpg')}).status_code, 413)

    def test_unknown_paths(self):
        self.assertEqual(self.client.post('/api/sample/unknown').status_code, 404)
        self.assertEqual(self.client.get('/assets/../../README.md').status_code, 404)

    def test_guidance_separates_predictions_from_other_diseases(self):
        result = self.client.get('/api/diseases').json
        catalog = {item['id']: item for item in result['diseases']}
        self.assertEqual(len(result['predicted_classes']), 4)
        for name in ['bunchy_top', 'bacterial_wilt', 'fusarium_wilt']:
            self.assertFalse(catalog[name]['model_supported'])
        self.assertTrue(catalog['fusarium_wilt']['experimental_available'])
        self.assertNotEqual(catalog['moko_wilt']['name'], catalog['bacterial_wilt']['name'])

    def test_whole_plant_symptoms_get_extension_review(self):
        result = self.client.post('/api/field-check', json={'symptoms': ['whole_plant_wilt']}).json
        self.assertEqual(result['priority'], 'extension_review')
        self.assertIn('even if the leaf model says Healthy', result['message'])
        self.assertIn('Rule-based', result['basis'])
        self.assertNotIn('predicted_class', result)
        self.assertEqual(self.client.post('/api/field-check', json={'symptoms': ['made_up']}).status_code, 400)

    def test_expanded_model_is_separate_and_real(self):
        models = self.client.get('/api/models').json['models']
        self.assertEqual(len(models['baseline']['class_names']), 4)
        self.assertEqual(len(models['expanded']['class_names']), 6)
        self.assertTrue(models['expanded']['experimental'])
        self.assertNotIn('bunchy_top', models['expanded']['class_names'])
        for name in ['fusarium_wilt', 'moko_wilt']:
            result = self.client.post('/api/sample/' + name + '?model=expanded').json
            self.assertEqual(result['predicted_class'], name)
            self.assertEqual(len(result['probabilities']), 6)
            self.assertTrue(result['experimental'])
            self.assertEqual(result['farmer_guidance']['id'], name)

    def test_invalid_model_does_not_fall_back(self):
        self.assertEqual(self.client.post('/api/predict', data={'model': 'unknown'}).status_code, 400)
        self.assertEqual(self.client.post('/api/sample/healthy?model=unknown').status_code, 400)

if __name__ == '__main__':
    unittest.main(verbosity=2)
