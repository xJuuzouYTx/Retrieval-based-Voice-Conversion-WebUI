import subprocess
import os
import shutil
from mega import Mega
import datetime

def find_parent(search_dir, file_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if file_name in filenames:
            return os.path.abspath(dirpath)
    return None

def find_folder_parent(search_dir, folder_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if folder_name in dirnames:
            return os.path.abspath(dirpath)
    return None

def download_from_url(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    
    if url != '':
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None
            
            if file_id:
                os.chdir('./zips')
                subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "--fuzzy"])
                
        elif "mega.nz" in url:
            if "#!" in url:
                file_id = url.split("#!")[1].split("!")[0]
            elif "file/" in url:
                file_id = url.split("file/")[1].split("/")[0]
            else:
                return None
            if file_id:
                m = Mega()
                m.download_url(url, zips_path)
        else:
            os.chdir('./zips')
            subprocess.run(["wget", url])
            
        os.chdir(parent_path)
        return "downloaded"
    else:
        return None
                

def load_downloaded_model(url):
    logs_folders = ['0_gt_wavs','1_16k_wavs','2a_f0','2b-f0nsf','3_feature256','3_feature768']
     
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    unzips_path = os.path.join(parent_path, 'unzips')
    weights_path = os.path.join(parent_path, 'weights')
    logs_dir = ""
    
    if os.path.exists(zips_path):
        shutil.rmtree(zips_path)
    if os.path.exists(unzips_path):
        shutil.rmtree(unzips_path)

    os.mkdir(zips_path)
    os.mkdir(unzips_path)
    
    download_file = download_from_url(url)
    if not download_file:
        return "No se ha podido descargar el modelo."
    # Descomprimir archivos descargados
    for filename in os.listdir(zips_path):
        if filename.endswith(".zip"):
            zipfile_path = os.path.join(zips_path,filename)
            shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
            logs_dir =  os.path.basename(os.path.join(parent_path,'logs', os.path.normpath(str(zipfile_path).replace(".zip",""))))
        else:
            return "El modelo se ha descargado pero no se ha podido descomprimir."
    
    index_file = False
    model_file = False
    D_file = False
    G_file = False
      
    # Copiar archivo pth
    for path, subdirs, files in os.walk(unzips_path):
        for item in files:
            item_path = os.path.join(path, item)
            if not 'G_' in item and not 'D_' in item and item.endswith('.pth'):
                model_file = True
                model_name = item.replace(".pth","")
                logs_dir = os.path.join(parent_path,'logs', model_name)
                if os.path.exists(logs_dir):
                    shutil.rmtree(logs_dir)
                os.mkdir(logs_dir)
                if not os.path.exists(weights_path):
                    os.mkdir(weights_path)
                if os.path.exists(os.path.join(weights_path, item)):
                    os.remove(os.path.join(weights_path, item))
                if os.path.exists(item_path):
                    shutil.move(item_path, weights_path)
    
    # Copiar index, D y G
    for path, subdirs, files in os.walk(unzips_path):
        for item in files:
            item_path = os.path.join(path, item)
            if item.startswith('added_') and item.endswith('.index'):
                index_file = True
                if os.path.exists(item_path):
                    shutil.move(item_path, logs_dir)
            if 'D_' in item and item.endswith('.pth'):
                D_file = True
                if os.path.exists(item_path):
                    shutil.move(item_path, logs_dir)
            if 'G_' in item and item.endswith('.pth'):
                G_file = True
                if os.path.exists(item_path):
                    shutil.move(item_path, logs_dir)
            if item.startswith('total_fea.npy') or item.startswith('events.'):
                if os.path.exists(item_path):
                    shutil.move(item_path, logs_dir)
            
    # Mover todos los folders excepto 'eval'
    for path, subdirs, files in os.walk(unzips_path):
        for folder in subdirs:
          if folder in logs_folders:
            item_path = os.path.join(path, folder)
            shutil.move(item_path, logs_dir)
            
    result = ""
    if model_file:
        if index_file:
            result += "El modelo funciona para inferencia, y tiene el archivo .index."
        else:
            result += "El modelo funciona para inferencia, pero no tiene el archivo .index."
    if D_file and G_file:
        if result:
            result += "\n"
        result += "El modelo puede ser reentrenado."
        
    return result

    # url = url.strip()
    # if url == '':
    #     return "La URL no puede estar vacia."
    # zip_dirs = ["zips", "unzips"]
    # for directory in zip_dirs:
    #     if os.path.exists(directory):
    #         shutil.rmtree(directory)
    # os.makedirs("zips", exist_ok=True)
    # os.makedirs("unzips", exist_ok=True)
    # zipfile = ''
    # zipfile_path = './zips/' + zipfile
    # MODELEPOCH = ''
    # print(url)
    # if "drive.google.com" in url:
    #     try:
    #         subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
    #     except Exception as e:
    #         try:
    #             print(e)
    #         except:
    #             return "Error al descargar"
    # elif "mega.nz" in url:
    #     m = Mega()
    #     m.download_url(url, './zips')
    # else:
    #     subprocess.run(["wget", url, "-O", f"./zips/{zipfile}"])
    # ---
    # for root, dirs, files in os.walk('./unzips'):
    #     for file in files:
    #         if "G_" in file:
    #             MODELEPOCH = file.split("G_")[1].split(".")[0]
    #     if MODELEPOCH == '':
    #         MODELEPOCH = '404'
    #     for file in files:
    #         file_path = os.path.join(root, file)
    #         if file.endswith(".npy") or file.endswith(".index"):
    #             subprocess.run(["mkdir", "-p", f"./logs/{model}"])
    #             subprocess.run(["mv", file_path, f"./logs/{model}/"])
    #         elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
    #             subprocess.run(["mv", file_path, f"./weights/{model}.pth"])
    # shutil.rmtree("zips")
    # shutil.rmtree("unzips")
    # return "Success."

def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_name

def save_to_wav2(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'./audios')
    return os.path.basename(file_path)

def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3')):
            audio_files.append(filename)
    return {"choices": sorted(audio_files), "__type__": "update"}
