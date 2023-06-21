import subprocess
import os
import shutil
from mega import Mega
import datetime
import unicodedata
import glob
import gradio as gr
import gdown
import zipfile

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

def get_drive_folder_id(url):
    if "drive.google.com" in url:
        if "file/d/" in url:
            file_id = url.split("file/d/")[1].split("/")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        else:
            return None

def download_from_url(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    
    if url != '':
        print(f"Descargando archivo: {url}")
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None
            
            if file_id:
                os.chdir('./zips')
                result = subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "--fuzzy"], capture_output=True, text=True)
                if "Too many users have viewed or downloaded this file recently" in str(result.stderr):
                    return "demasiado uso"
                print(result.stderr)
                
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
        print("Descarga completa.")
        return "downloaded"
    else:
        return None
                
class error_message(Exception):
    def __init__(self, mensaje):
        self.mensaje = mensaje
        super().__init__(mensaje)
        
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
        elif download_file == "downloaded":
            print("Modelo descargado correctamente. Procediendo con la extracción...")
            infos.append("Modelo descargado correctamente. Procediendo con la extracción...")
            yield "\n".join(infos)
        elif download_file == "demasiado uso":
            raise Exception("demasiado uso")
        
        # Descomprimir archivos descargados
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                shutil.unpack_archive(zipfile_path, unzips_path, 'zip')
                model_name = os.path.basename(zipfile_path)
                logs_dir = os.path.join(parent_path,'logs', os.path.normpath(str(model_name).replace(".zip","")))
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
        
        if not model_file and not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        # Copiar index, D y G
        for path, subdirs, files in os.walk(unzips_path):
            for item in files:
                item_path = os.path.join(path, item)
                if item.startswith('added_') and item.endswith('.index'):
                    index_file = True
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
                if 'D_' in item and item.endswith('.pth'):
                    D_file = True
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
                if 'G_' in item and item.endswith('.pth'):
                    G_file = True
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
                        shutil.move(item_path, logs_dir)
                if item.startswith('total_fea.npy') or item.startswith('events.'):
                    if os.path.exists(item_path):
                        if os.path.exists(os.path.join(logs_dir, item)):
                            os.remove(os.path.join(logs_dir, item))
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
                print("El modelo funciona para inferencia, y tiene el archivo .index.")
                infos.append("\nEl modelo funciona para inferencia, y tiene el archivo .index.")
                yield "\n".join(infos)
            else:
                print("El modelo funciona para inferencia, pero no tiene el archivo .index.")
                infos.append("\nEl modelo funciona para inferencia, pero no tiene el archivo .index.")
                yield "\n".join(infos)
        if D_file and G_file:
            if result:
                result += "\n"
            print("El modelo puede ser reentrenado.")
            infos.append("El modelo puede ser reentrenado.")
            yield "\n".join(infos)
        
        os.chdir(parent_path)    
        return result
    except Exception as e:
        os.chdir(parent_path)
        if "demasiado uso" in str(e):
            print("Demasiados usuarios han visto o descargado este archivo recientemente. Por favor, intenta acceder al archivo nuevamente más tarde. Si el archivo al que estás intentando acceder es especialmente grande o está compartido con muchas personas, puede tomar hasta 24 horas para poder ver o descargar el archivo. Si aún no puedes acceder al archivo después de 24 horas, ponte en contacto con el administrador de tu dominio.")
            yield "El enlace llegó al limite de uso, intenta nuevamente más tarde o usa otro enlace."    
        else:
            print(e)
            yield "Ocurrio un error descargando el modelo"
    finally:
        os.chdir(parent_path)
      
def load_dowloaded_dataset(url):
    parent_path = find_folder_parent(".", "pretrained_v2")
    infos = []
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
            raise Exception("No se ha podido descargar el dataset.")
        else:
            infos.append("Dataset descargado. Procediendo con la extracción...")
            yield "\n".join(infos)

        zip_path = os.listdir(zips_path)
        foldername = ""
        for file in zip_path:
            if file.endswith('.zip'):
                file_path = os.path.join(zips_path, file)
                with zipfile.ZipFile(file_path, "r") as archivo_zip:
                    lista_archivos = archivo_zip.namelist()
                    if lista_archivos[0].endswith('/') and any(f.startswith(lista_archivos[0]) for f in lista_archivos[1:]):
                        print("El archivo ZIP contiene un solo directorio y todos los archivos están dentro de ese directorio.")
                        foldername = lista_archivos[0].replace("/","")
                    else:
                        print("El archivo ZIP fue comprimido fuera de un folder. Intentando proceder con la extracción....")
                        foldername = file.replace(".zip","").replace(" ","").replace("-","_")
                        datasets_path = os.path.join(datasets_path, foldername)

                shutil.unpack_archive(file_path, datasets_path, 'zip')
                
                datasets_path = os.path.join(parent_path, 'datasets')
                # Renombrar folder en /rvc/datatasets si tiene espacios
                new_dataset_folder_name = os.path.join(datasets_path, foldername).replace(" ","").replace("(","").replace(")","").replace("-","_")
                os.rename(os.path.join(datasets_path, foldername), new_dataset_folder_name)
                
                new_name = file_path.replace(" ", "").encode("ascii", "ignore").decode()
                os.rename(file_path, new_name)
        
        infos.append("Dataset cargado correctamente.")
        yield "\n".join(infos)
    except Exception as e:
        os.chdir(parent_path)
        print(e)
        if "No se ha podido descargar el dataset." in str(e):
            infos.append("Ocurrio un error al descargar, intentalo de nuevo o usa otro enlace.")
            yield "\n".join(infos)
        else:
            infos.append("Ocurrio un error desconocido.")
            yield "\n".join(infos)

def save_model(modelname, save_action):
       
    parent_path = find_folder_parent(".", "pretrained_v2")
    zips_path = os.path.join(parent_path, 'zips')
    dst = os.path.join(zips_path,modelname)
    logs_path = os.path.join(parent_path, 'logs', modelname)
    weights_path = os.path.join(parent_path, 'weights', f"{modelname}.pth")
    save_folder = parent_path
    infos = []    
    
    try:
        if not os.path.exists(logs_path):
            raise Exception("No model found.")
        
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
            print("Guardar todo")
            shutil.copytree(logs_path, dst)
        else:
            # Si no existe el folder donde se va a comprimir el modelo
            if not os.path.exists(dst):
                os.mkdir(dst)
            
        if save_action == "Guardar D y G":
            print("Guardar D y G")
            if len(d_file) > 0:
                shutil.copy(d_file[0], dst)
            if len(g_file) > 0:
                shutil.copy(g_file[0], dst)    
                
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append("Guardando sin indice...")
                
        if save_action == "Guardar voz":
            print("Guardar Voz")
            if len(added_file) > 0:
                shutil.copy(added_file[0], dst)
            else:
                infos.append("Guardando sin indice...")
                #raise gr.Error("¡No ha generado el archivo added_*.index!")
        
        yield "\n".join(infos)
        # Si no existe el archivo del modelo no copiarlo
        if not os.path.exists(weights_path):
            infos.append("Guardando sin modelo pequeño...")
            #raise gr.Error("¡No ha generado el modelo pequeño!")
        else:
            shutil.copy(weights_path, dst)
        
        yield "\n".join(infos)
        infos.append("\nEsto puede tomar unos minutos, por favor espere...")
        yield "\n".join(infos)
        
        shutil.make_archive(os.path.join(zips_path,f"{modelname}"), 'zip', zips_path)
        shutil.move(os.path.join(zips_path,f"{modelname}.zip"), os.path.join(save_folder, f'{modelname}.zip'))
        
        shutil.rmtree(zips_path)
        #shutil.rmtree(zips_path)
        
        infos.append("\n¡Modelo guardado!")
        yield "\n".join(infos)
        
    except Exception as e:
        print(e)
        if "No model found." in str(e):
            infos.append("El modelo que intenta guardar no existe, asegurese de escribir el nombre correctamente.")
        else:
            infos.append("Ocurrio un error guardando el modelo")
            
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
        print("********")
        print(e)
        print("********")
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
