import os
from pathlib import Path

print('--- Environment variables (relevant) ---')
for k in ['HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME', 'HF_DATASETS_CACHE']:
    print(f"{k} = {os.environ.get(k)!r}")

home = Path.home()

candidates = [
    Path(os.environ.get('HF_HOME')) if os.environ.get('HF_HOME') else None,
    Path(os.environ.get('HUGGINGFACE_HUB_CACHE')) if os.environ.get('HUGGINGFACE_HUB_CACHE') else None,
    Path(os.environ.get('TRANSFORMERS_CACHE')) if os.environ.get('TRANSFORMERS_CACHE') else None,
    Path(os.environ.get('TORCH_HOME')) if os.environ.get('TORCH_HOME') else None,
    Path(os.environ.get('HF_DATASETS_CACHE')) if os.environ.get('HF_DATASETS_CACHE') else None,
    home / '.cache' / 'huggingface',
    home / '.cache' / 'huggingface' / 'hub',
    home / '.cache' / 'transformers',
    home / '.cache' / 'torch',
    home / 'AppData' / 'Local' / 'huggingface',
    home / 'AppData' / 'Local' / 'huggingface' / 'hub',
]

seen = set()
print('\n--- Candidate cache directories and sample contents ---')
for p in candidates:
    if not p:
        continue
    if str(p) in seen:
        continue
    seen.add(str(p))
    print('\nPath:', p)
    print('Exists:', p.exists())
    try:
        if p.exists() and p.is_dir():
            entries = list(p.iterdir())
            print('First 20 entries:')
            for e in entries[:20]:
                try:
                    size = e.stat().st_size
                except Exception:
                    size = 'n/a'
                print(' -', e.name, '(dir)' if e.is_dir() else f'({size} bytes)')
        else:
            # if it's a file, print info
            if p.exists():
                print('File size:', p.stat().st_size)
    except PermissionError:
        print('Permission denied when listing', p)

print('\n--- Search for model folders under home cache (huggingface/diffusers/transformers/torch) ---')
search_roots = [home / '.cache' / 'huggingface', home / '.cache' / 'torch', home / '.cache' / 'transformers']
for root in search_roots:
    print('\nRoot:', root)
    if root.exists() and root.is_dir():
        found = False
        for sub in root.rglob('*'):
            # show directories that look like model folders (contain 'pytorch_model' or 'diffusers' names or '.bin' files)
            if sub.is_dir() and any(x in sub.name.lower() for x in ('diffusers', 'models', 'checkpoints')):
                print(' - possible model dir:', sub)
                found = True
                break
            if sub.is_file() and sub.suffix in ('.bin', '.safetensors'):
                print(' - found model file:', sub)
                found = True
                break
        if not found:
            print(' - no obvious model files found under', root)
    else:
        print(' - root does not exist')

print('\nScript finished')
