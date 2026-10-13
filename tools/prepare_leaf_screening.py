"""Download the pinned image-screening model once; later inference is offline."""
import os
os.environ['USE_TF'] = '0'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import requests
from transformers import CLIPModel, CLIPProcessor

folder = Path(__file__).resolve().parents[1] / 'output/leaf_screening_model'
folder.mkdir(parents=True, exist_ok=True)
if (folder / 'ready.txt').is_file():
    print('Screening model already prepared.')
    raise SystemExit(0)
revision = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
base = f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{revision}/'
parts = folder / 'download_parts'
parts.mkdir(exist_ok=True)
total, block = 605247071, 1024 * 1024
def download_part(start):
    end = min(total, start + block) - 1
    path = parts / str(start)
    if path.is_file() and path.stat().st_size == end - start + 1:
        return path
    for attempt in range(8):
        try:
            response = requests.get(base + 'pytorch_model.bin?part=' + str(start), headers={'Range': f'bytes={start}-{end}'}, timeout=90)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise ValueError('Incomplete model part')
            path.write_bytes(response.content)
            if start // block % 25 == 0:
                print('Downloaded part', start // block + 1, '/ 578', flush=True)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 20))
print('Preparing local screening model...', flush=True)
weights = folder / 'pytorch_model.bin'
if not weights.is_file():
    with ThreadPoolExecutor(max_workers=24) as pool:
        downloaded = list(pool.map(download_part, range(0, total, block)))
    with weights.open('wb') as file:
        for path in downloaded:
            file.write(path.read_bytes())
with weights.open('rb') as file:
    checksum = hashlib.file_digest(file, 'sha256').hexdigest()
assert checksum == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f', 'Model checksum mismatch'
for name in ['config.json', 'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
    path = folder / name
    if not path.is_file():
        response = requests.get(base + name, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
model = CLIPModel.from_pretrained(folder, local_files_only=True, use_safetensors=False)
processor = CLIPProcessor.from_pretrained(folder, local_files_only=True)
model.save_pretrained(folder)
processor.save_pretrained(folder)
(folder / 'ready.txt').write_text('Local model and processor prepared successfully.\n')
weights.unlink()
for path in parts.iterdir():
    path.unlink()
parts.rmdir()
print('Screening model restored and checksum verified.', flush=True)
