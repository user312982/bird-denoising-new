import json

notebook_path = '/home/izanagi/Documents/Code/bird-denoising-new/notebook.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_source = [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Membuat struktur direktori lokal untuk menampung data\n",
    "os.makedirs('data/train/noisy', exist_ok=True)\n",
    "os.makedirs('data/train/clean', exist_ok=True)\n",
    "os.makedirs('data/valid/noisy', exist_ok=True)\n",
    "os.makedirs('data/valid/clean', exist_ok=True)\n",
    "os.makedirs('data/test/noisy', exist_ok=True)\n",
    "os.makedirs('data/test/clean', exist_ok=True)\n",
    "os.makedirs('checkpoints', exist_ok=True)\n",
    "\n",
    "def copy_600_files(src_noisy, src_clean, dst_noisy, dst_clean):\n",
    "    try:\n",
    "        # Ambil 600 file berformat .wav yang ada di folder clean\n",
    "        files = sorted([f for f in os.listdir(src_clean) if f.endswith('.wav')])[:600]\n",
    "        for f in files:\n",
    "            shutil.copy(os.path.join(src_noisy, f), dst_noisy)\n",
    "            shutil.copy(os.path.join(src_clean, f), dst_clean)\n",
    "        print(f\"Berhasil menyalin {len(files)} file ke {dst_clean} dan {dst_noisy}\")\n",
    "    except FileNotFoundError as e:\n",
    "        print(f\"Direktori tidak ditemukan: {e}\")\n",
    "\n",
    "# Opsional: Mengambil 600 data saja dengan nama yang sama untuk train dan valid\n",
    "# Menyalin untuk data train\n",
    "copy_600_files(\n",
    "    '/content/drive/MyDrive/dataset/unzip/Raw_audios',      # Raw -> train/noisy\n",
    "    '/content/drive/MyDrive/dataset/unzip/Denoised_audios', # Denoised -> train/clean\n",
    "    'data/train/noisy',\n",
    "    'data/train/clean'\n",
    ")\n",
    "\n",
    "# Menyalin untuk data valid\n",
    "copy_600_files(\n",
    "    '/content/drive/MyDrive/dataset/unzip/valid/Raw_audios',        # Raw -> valid/noisy\n",
    "    '/content/drive/MyDrive/dataset/unzip/valid/Denoised_audios',   # Denoised -> valid/clean\n",
    "    'data/valid/noisy',\n",
    "    'data/valid/clean'\n",
    ")\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        if "!cp -r /content/drive/MyDrive/dataset/unzip/Denoised_audios/*.wav data/train/noisy/" in source_str:
            cell['source'] = new_source
            print("Updated cell")
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated.")
