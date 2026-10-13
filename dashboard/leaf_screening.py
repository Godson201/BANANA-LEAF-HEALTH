"""Heuristic photo-subject screen; not a validated botanical species detector."""
import os
os.environ.setdefault('USE_TF', '0')
from pathlib import Path
from functools import lru_cache
import threading
import torch

MODEL_FOLDER = Path(__file__).resolve().parents[1] / 'output/leaf_screening_model'
BANANA_PROMPTS = ['a close-up photograph of a banana plant leaf',
                  'a photograph of a diseased banana leaf with brown spots',
                  'a photograph of a broad banana leaf with parallel veins']
OTHER_PROMPTS = ['a photograph of a mango tree leaf', 'a photograph of a maize corn leaf',
    'a photograph of a tomato plant leaf', 'a photograph of a bean plant leaf',
    'a photograph of an oak tree leaf', 'a photograph of a maple tree leaf',
    'a photograph of a palm frond', 'a photograph of grass', 'a photograph of a flower',
    'a photograph of a plant with flowers', 'a photograph of a person', 'a photograph of an animal',
    'a photograph of a car', 'a photograph of a building', 'a photograph of food or fruit',
    'a screenshot of text or a document', 'a photograph of an object indoors', 'a drawing or illustration of a leaf']
lock = threading.Lock()
class ScreeningUnavailable(RuntimeError):
    pass
def screening_ready():
    return (MODEL_FOLDER / 'ready.txt').is_file()
@lru_cache(maxsize=1)
def load_screening_model():
    if not screening_ready():
        raise ScreeningUnavailable('Prepare the screening model first.')
    from transformers import CLIPModel, CLIPProcessor
    try:
        model = CLIPModel.from_pretrained(MODEL_FOLDER, local_files_only=True).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_FOLDER, local_files_only=True)
        tokens = processor(text=BANANA_PROMPTS + OTHER_PROMPTS, return_tensors='pt', padding=True)
        with torch.inference_mode():
            text = model.text_projection(model.text_model(**tokens).pooler_output)
            text = torch.nn.functional.normalize(text, dim=-1)
        return model, processor, text
    except (OSError, ValueError, RuntimeError) as error:
        raise ScreeningUnavailable('Screening model could not load.') from error
def screen_photo(photo):
    if min(photo.size) < 128:
        return {'accepted': False, 'reason': 'too_small', 'message': 'Use a clear photo at least 128 pixels on each side.'}
    model, processor, text = load_screening_model()
    inputs = processor(images=photo, return_tensors='pt')
    with lock, torch.inference_mode():
        features = model.visual_projection(model.vision_model(pixel_values=inputs['pixel_values']).pooler_output)
        similarities = (torch.nn.functional.normalize(features, dim=-1) @ text.T)[0]
    banana = float(similarities[:len(BANANA_PROMPTS)].max())
    other = float(similarities[len(BANANA_PROMPTS):].max())
    accepted = banana >= .23 and banana - other >= .015
    return {'accepted': accepted, 'reason': 'suitable' if accepted else 'subject_not_confirmed',
            'message': 'Photo passed the subject check.' if accepted else 'Unable to confirm a banana leaf. This may be another plant, an unrelated image or an unclear photo.',
            'method': 'CLIP image-text similarity heuristic; not a species guarantee',
            'banana_similarity': round(banana, 4), 'alternative_similarity': round(other, 4), 'margin': round(banana - other, 4)}
