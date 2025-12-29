import os
import requests
from tqdm import tqdm  # Import library progress bar

# Konfigurasi
START_YEAR = 2008
END_YEAR = 2013
OUTPUT_DIR = "data_sst"

# Buat folder jika belum ada
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Memulai download ke folder: {OUTPUT_DIR}\n")

for year in range(START_YEAR, END_YEAR + 1):
    # URL file NOAA OISST V2 High Res (Daily Mean)
    file_name = f"sst.day.mean.{year}.nc"
    url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/{file_name}"
    
    save_path = os.path.join(OUTPUT_DIR, file_name)
    
    # Cek jika file sudah ada agar tidak download ulang
    if os.path.exists(save_path):
        print(f"File {file_name} sudah ada. Skip.")
        continue
        
    try:
        # Stream=True penting untuk download file besar agar header bisa dibaca dulu
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        
        # Ambil ukuran total file dari header (dalam bytes)
        total_size = int(response.headers.get('content-length', 0))
        
        # Setup Progress Bar
        # unit='B': satuan bytes
        # unit_scale=True: otomatis ubah ke KB, MB, GB
        # desc: Label nama file di sebelah progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, ascii=False) as pbar:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # Update progress bar sesuai ukuran chunk yang diterima
                        pbar.update(len(chunk))
        
    except Exception as e:
        print(f"\nGagal mendownload {file_name}: {e}")

print("\nSemua proses selesai!")
