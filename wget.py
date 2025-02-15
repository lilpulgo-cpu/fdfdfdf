import os
import shutil
import tarfile
import wget

# Crear los directorios de destino si no existen
os.makedirs("models", exist_ok=True)
os.makedirs("data/avatars", exist_ok=True)

# ================================
# Descarga y copia del primer archivo
# ================================

# URL del primer archivo (wav2lip384.pth)
url1 = ("https://drive.usercontent.google.com/download?"
        "id=10nOKATfuaxsmDU8sG2dQkQyq68KLslvZ&export=download&authuser=0&confirm=t&"
        "uuid=81dce160-e018-4e35-8db0-dd3c5926ab62&at=AIrpjvP5eJw6nrTkmuSu8S5iIJkW:1739582743480")
nombre_archivo1 = "wav2lip384.pth"

print("Descargando wav2lip384.pth...")
wget.download(url1, nombre_archivo1)
print("\nDescarga completada.")

# Copiar el archivo descargado a la carpeta models y renombrarlo a wav2lip.pth
destino_model = os.path.join("models", "wav2lip.pth")
shutil.copy(nombre_archivo1, destino_model)
print(f"Archivo copiado y renombrado a: {destino_model}")

# ================================
# Descarga y extracción del segundo archivo
# ================================

# URL del segundo archivo (wav2lip384_avatar1.tar.gz)
url2 = ("https://drive.usercontent.google.com/download?"
        "id=1RRWbPEvxNzPbuplIqDrhrZZR8WzBlFVv&export=download&authuser=0&confirm=t&"
        "uuid=76191218-ce67-4568-bae0-b63705301eb7&at=AIrpjvPq7iI5yCHAaISE9g2SeZAN:1739582732553")
nombre_archivo2 = "wav2lip384_avatar1.tar.gz"

print("Descargando wav2lip384_avatar1.tar.gz...")
wget.download(url2, nombre_archivo2)
print("\nDescarga completada.")

# Extraer el archivo tar.gz a un directorio temporal
directorio_temp = "temp_extracted"
os.makedirs(directorio_temp, exist_ok=True)

with tarfile.open(nombre_archivo2, "r:gz") as tar:
    tar.extractall(path=directorio_temp)
print("Archivo tar.gz extraído.")

# Copiar la carpeta extraída a data/avatars
# Suponiendo que el tar.gz contiene una carpeta principal o varios elementos
for item in os.listdir(directorio_temp):
    origen = os.path.join(directorio_temp, item)
    destino = os.path.join("data/avatars", item)
    if os.path.isdir(origen):
        # Copia recursiva de la carpeta (dirs_exist_ok requiere Python 3.8+)
        shutil.copytree(origen, destino, dirs_exist_ok=True)
    else:
        shutil.copy2(origen, destino)
print("Carpeta(s) extraída(s) copiada(s) a data/avatars.")

# Eliminar el directorio temporal de extracción
shutil.rmtree(directorio_temp)
print("Directorio temporal eliminado.")
