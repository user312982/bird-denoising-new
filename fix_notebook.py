import json

notebook_path = 'notebook.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if "Denoised_audios', # Denoised -> train/noisy" in line:
                new_source.append("    '/content/drive/MyDrive/dataset/unzip/Raw_audios',      # Raw -> train/noisy\n")
            elif "Raw_audios',      # Raw -> train/clean" in line:
                new_source.append("    '/content/drive/MyDrive/dataset/unzip/Denoised_audios', # Denoised -> train/clean\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook paths fixed.")
