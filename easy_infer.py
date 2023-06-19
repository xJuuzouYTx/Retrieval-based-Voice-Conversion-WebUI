import subprocess
import os
import shutil
from mega import Mega
import datetime
import unicodedata
import glob
import gradio as gr
import gdown

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
    parent_path = find_folder_parent(".", "pretrained_v2")
    try:
        infos = []
        logs_folders = ['0_gt_wavs','1_16k_wavs','2a_f0','2b-f0nsf','3_feature256','3_feature768']
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
            print("No se ha podido descargar el modelo.")
            infos.append("No se ha podido descargar el modelo.")
            yield "\n".join(infos)
        else:
            print("Modelo descargado correctamente. Procediendo con la extracción...")
            infos.append("Modelo descargado correctamente. Procediendo con la extracción...")
            yield "\n".join(infos)
            
        # Descomprimir archivos descargados
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
                logs_dir = os.path.join(parent_path,'logs', os.path.normpath(str(zipfile_path).replace(".zip","")))
                print("Modelo descomprimido correctamente. Copiando a logs...")
                infos.append("Modelo descomprimido correctamente. Copiando a logs...")
                yield "\n".join(infos)
            else:
                print("Error al descomprimir el modelo.")
                infos.append("Error al descomprimir el modelo.")
                yield "\n".join(infos)
        
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
                infos.append("\nEl modelo funciona para inferencia, y tiene el archivo .index.")
                yield "\n".join(infos)
            else:
                infos.append("\nEl modelo funciona para inferencia, pero no tiene el archivo .index.")
                yield "\n".join(infos)
        if D_file and G_file:
            if result:
                result += "\n"
            infos.append("\nEl modelo puede ser reentrenado.")
            yield "\n".join(infos)
        
        os.chdir(parent_path)    
        return result
    except Exception as e:
        os.chdir(parent_path)
        print(e)
        return "Ocurrio un error descargando el modelo"
    finally:
        os.chdir(parent_path)
      
def load_dowloaded_dataset(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    
    try:
        zips_path = os.path.join(parent_path, 'zips')
        unzips_path = os.path.join(parent_path, 'unzips')
        datasets_path = os.path.join(parent_path, 'datasets')
        
        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
        if not os.path.exists(datasets_path):
            os.mkdir(datasets_path)
            
        os.mkdir(zips_path)
        os.mkdir(unzips_path)
        
        download_file = download_from_url(url)
        if not download_file:
            return "No se ha podido descargar el dataset."

        zip_path = os.listdir(zips_path)
        for file in zip_path:
            if file.endswith('.zip'):
                file_path = os.path.join(zips_path, file)
                shutil.unpack_archive(file_path, datasets_path, 'zip')
                new_name = file_path.replace(" ", "").encode("ascii", "ignore").decode()
                os.rename(file_path, new_name)
                
        return "Dataset descargado."
    except Exception as e:
        os.chdir(parent_path)
        print(e)
        return "Error al descargar"

def save_model(modelname, save_action):
    infos = []
    
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    dst = os.path.join(zips_path,modelname)
    logs_path = os.path.join(parent_path, 'logs', modelname)
    weights_path = os.path.join(parent_path, 'weights', f"{modelname}.pth")
    save_folder = parent_path
    
    if not 'content' in parent_path:
        save_folder = os.path.join(parent_path, 'RVC')
    else:
        save_folder = '/content/drive/MyDrive/RVC'
    
    infos.append(f"Guardando modelo en: {save_folder}")
    yield "\n".join(infos)
    
    # Si no existe el folder RVC para guardar los modelos
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    # Si ya existe el folders zips borro su contenido por si acaso
    if os.path.exists(zips_path):
        shutil.rmtree(zips_path)
        
    os.mkdir(zips_path)
    
    added_file = glob.glob(os.path.join(logs_path, "added_*.index"))
    d_file = glob.glob(os.path.join(logs_path, "D_*.pth"))
    g_file = glob.glob(os.path.join(logs_path, "G_*.pth"))
    
    if save_action == "Guardar todo":
        shutil.copytree(logs_path, dst)
    else:
        # Si no existe el folder donde se va a comprimir el modelo
        if not os.path.exists(dst):
            os.mkdir(dst)
        
    if save_action == "Guardar D y G":

        if len(d_file) > 0:
            shutil.copy(d_file[0], dst)
        if len(g_file) > 0:
            shutil.copy(g_file[0], dst)    
        if len(added_file) > 0:
            shutil.copy(added_file[0], dst)
            
    if save_action == "Guardar voz":
        pass
        if len(added_file) > 0:
            shutil.copy(added_file[0], dst)
        else:
            shutil.rmtree(zips_path)
            raise gr.Error("¡No ha generado el archivo added_*.index!")
    
    yield "\n".join(infos)
    # Si no existe el archivo del modelo no copiarlo
    if not os.path.exists(weights_path):
        shutil.rmtree(zips_path)
        raise gr.Error("¡No ha generado el modelo pequeño!")
    else:
        shutil.copy(weights_path, dst)
    
    yield "\n".join(infos)
    infos.append("\nEsto puede tomar unos minutos, por favor espere...")    
    infos.append("\nComprimiendo modelo...")
    yield "\n".join(infos)
    
    shutil.make_archive(os.path.join(zips_path,f"{modelname}"), 'zip', zips_path)
    shutil.move(os.path.join(zips_path,f"{modelname}.zip"), os.path.join(save_folder, f'{modelname}.zip'))
    
    shutil.rmtree(zips_path)
    #shutil.rmtree(zips_path)
    
    infos.append("\n¡Modelo guardado!")
    yield "\n".join(infos)
    
def load_dowloaded_backup(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    logs_path = os.path.join(parent_path, 'logs')
    infos = []
    print("Descargando en...")
    infos.append(f"\nDescargando en {logs_path}")
    yield "\n".join(infos)
    try:
        os.chdir(logs_path)
        filename = gdown.download_folder(url=url,quiet=True, remaining_ok=True)
        infos.append(f"\nBackup cargado: {filename}")
        yield "\n".join(infos)
    except Exception as e:
        print(e)
        infos.append("\nOcurrió un error al descargar")
        yield "\n".join(infos)
    finally:
        os.chdir(parent_path)

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
