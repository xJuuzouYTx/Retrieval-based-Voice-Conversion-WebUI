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
import json
import requests

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
                result = subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "--fuzzy"], capture_output=True, text=True, encoding='utf-8')
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
                print("Intentando proceder con la extracción....")
                foldername = file.replace(".zip","").replace(" ","").replace("-","_")
                dataset_path = os.path.join(datasets_path, foldername)
                shutil.unpack_archive(file_path, unzips_path, 'zip')
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)
                    
                os.mkdir(dataset_path)
                
                for root, subfolders, songs in os.walk(unzips_path):
                    for song in songs:
                        song_path = os.path.join(root, song)
                        if song.endswith(".wav"):
                            shutil.move(song_path, dataset_path)

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)
            
        infos.append(f"Dataset cargado correctamente.")
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

def get_models_by_name(modelname):
    url = "https://script.google.com/macros/s/AKfycbwPiL_l8Q1jczJiqDVIKMqRLoocVWuLCP1fKgv0T8nDszvVcD5s8SCYnrWfqM4z5barMA/exec"
    
    response = requests.post(url, json={
        'type': 'search_by_name',
        'name': modelname.strip().lower()
    })

    models = response.json()
    
    result = []
    message = "Busqueda realizada"
    if len(models) == 0:
        message = "No se han encontrado resultados."
    else:
        message = f"Se han encontrado {len(models)} resultados para {modelname}"
        
    for i in range(20):
        if i  < len(models):
            
            # Nombre
            result.append(
                {
                    "visible": True,
                    "value": str("### ") + str(models[i].get('name')),
                    "__type__": "update",
                })
            # Url
            result.append(
                {
                    "visible": False,
                    "value": models[i].get('url'),
                    "__type__": "update",
                })
            # Boton
            result.append({
                    "visible": True,
                    "__type__": "update",
                })
            
            # Linea separadora
            if i == len(models) - 1:
                result.append({
                        "visible": False,
                        "__type__": "update",
                })
            else:
                result.append({
                    "visible": True,
                    "__type__": "update",
                })
                
            # Row
            result.append(
                {
                    "visible": True,
                    "__type__": "update",
                })
        else:
            # Nombre
            result.append(
                {
                    "visible": False,
                    "__type__": "update",
                })
            # Url
            result.append(
                {
                    "visible": False,
                    "value": False,
                    "__type__": "update",
                })
            # Boton
            result.append({
                    "visible": False,
                    "__type__": "update",
                })
            # Linea
            result.append({
                    "visible": False,
                    "__type__": "update",
                })
            # Row
            result.append(
                {
                    "visible": False,
                    "__type__": "update",
                })
     # Result
    result.append(
        {
            "value": message,
            "__type__": "update",
        }
    )
    
    return result

def search_model():
    gr.Markdown(value="# Buscar un modelo")
    with gr.Row():
        model_name = gr.inputs.Textbox(lines=1, label="Término de búsqueda")
        search_model_button=gr.Button("Buscar modelo")
        
    models = []
    results = gr.Textbox(label="Resultado", value="", max_lines=20)
    with gr.Row(visible=False) as row1:
        l1 = gr.Markdown(value="", visible=False)
        l1_url = gr.Textbox("Label 1", visible=False)
        b1 = gr.Button("Cargar modelo", visible=False)
    
    mk1 = gr.Markdown(value="---", visible=False)
    b1.click(fn=load_downloaded_model, inputs=l1_url, outputs=results)
    
    with gr.Row(visible=False) as row2:
        l2 = gr.Markdown(value="", visible=False)
        l2_url = gr.Textbox("Label 1", visible=False)
        b2 = gr.Button("Cargar modelo", visible=False)
    
    mk2 = gr.Markdown(value="---", visible=False)
    b2.click(fn=load_downloaded_model, inputs=l2_url, outputs=results)
    
    with gr.Row(visible=False) as row3:
        l3 = gr.Markdown(value="", visible=False)
        l3_url = gr.Textbox("Label 1", visible=False)
        b3 = gr.Button("Cargar modelo", visible=False)
    
    mk3 = gr.Markdown(value="---", visible=False)
    b3.click(fn=load_downloaded_model, inputs=l3_url, outputs=results)
        
    with gr.Row(visible=False) as row4:
        l4 = gr.Markdown(value="", visible=False)
        l4_url = gr.Textbox("Label 1", visible=False)
        b4 = gr.Button("Cargar modelo", visible=False)
    mk4 = gr.Markdown(value="---", visible=False)    
    b4.click(fn=load_downloaded_model, inputs=l4_url, outputs=results)
    
    with gr.Row(visible=False) as row5:
        l5 = gr.Markdown(value="", visible=False)
        l5_url = gr.Textbox("Label 1", visible=False)
        b5 = gr.Button("Cargar modelo", visible=False)
    
    mk5 = gr.Markdown(value="---", visible=False) 
    b5.click(fn=load_downloaded_model, inputs=l5_url, outputs=results)
        
    with gr.Row(visible=False) as row6:
        l6 = gr.Markdown(value="", visible=False)
        l6_url = gr.Textbox("Label 1", visible=False)
        b6 = gr.Button("Cargar modelo", visible=False)

    mk6 = gr.Markdown(value="---", visible=False)        
    b6.click(fn=load_downloaded_model, inputs=l6_url, outputs=results)
        
    with gr.Row(visible=False) as row7:
        l7 = gr.Markdown(value="", visible=False)
        l7_url = gr.Textbox("Label 1", visible=False)
        b7 = gr.Button("Cargar modelo", visible=False)
    
    mk7 = gr.Markdown(value="---", visible=False)
    b7.click(fn=load_downloaded_model, inputs=l7_url, outputs=results)
        
    with gr.Row(visible=False) as row8:
        l8 = gr.Markdown(value="", visible=False)
        l8_url = gr.Textbox("Label 1", visible=False)
        b8 = gr.Button("Cargar modelo", visible=False)
    
    mk8 = gr.Markdown(value="---", visible=False)
    b8.click(fn=load_downloaded_model, inputs=l8_url, outputs=results)
        
    with gr.Row(visible=False) as row9:
        l9 = gr.Markdown(value="", visible=False)
        l9_url = gr.Textbox("Label 1", visible=False)
        b9 = gr.Button("Cargar modelo", visible=False)
        
    mk9 = gr.Markdown(value="---", visible=False)
    b9.click(fn=load_downloaded_model, inputs=l9_url, outputs=results)
        
    with gr.Row(visible=False) as row10:
        l10 = gr.Markdown(value="", visible=False)
        l10_url = gr.Textbox("Label 1", visible=False)
        b10 = gr.Button("Cargar modelo", visible=False)
    
    mk10 = gr.Markdown(value="---", visible=False) 
    b10.click(fn=load_downloaded_model, inputs=l10_url, outputs=results)
        
    with gr.Row(visible=False) as row11:
        l11 = gr.Markdown(value="", visible=False)
        l11_url = gr.Textbox("Label 1", visible=False)
        b11 = gr.Button("Cargar modelo", visible=False)
        
    mk11 = gr.Markdown(value="---", visible=False)
    b11.click(fn=load_downloaded_model, inputs=l11_url, outputs=results)
        
    with gr.Row(visible=False) as row12:
        l12 = gr.Markdown(value="", visible=False)
        l12_url = gr.Textbox("Label 1", visible=False)
        b12 = gr.Button("Cargar modelo", visible=False)

    mk12 = gr.Markdown(value="---", visible=False)        
    b12.click(fn=load_downloaded_model, inputs=l12_url, outputs=results)
        
    with gr.Row(visible=False) as row13:
        l13 = gr.Markdown(value="", visible=False)
        l13_url = gr.Textbox("Label 1", visible=False)
        b13 = gr.Button("Cargar modelo", visible=False)
    
    mk13 = gr.Markdown(value="---", visible=False)
    b13.click(fn=load_downloaded_model, inputs=l13_url, outputs=results)
    
    with gr.Row(visible=False) as row14:
        l14 = gr.Markdown(value="", visible=False)
        l14_url = gr.Textbox("Label 1", visible=False)
        b14 = gr.Button("Cargar modelo", visible=False)
    
    mk14 = gr.Markdown(value="---", visible=False)
    b14.click(fn=load_downloaded_model, inputs=l14_url, outputs=results)
    
    with gr.Row(visible=False) as row15:
        l15 = gr.Markdown(value="", visible=False)
        l15_url = gr.Textbox("Label 1", visible=False)
        b15 = gr.Button("Cargar modelo", visible=False)

    mk15 = gr.Markdown(value="---", visible=False)        
    b15.click(fn=load_downloaded_model, inputs=l15_url, outputs=results)
        
    with gr.Row(visible=False) as row16:
        l16 = gr.Markdown(value="", visible=False)
        l16_url = gr.Textbox("Label 1", visible=False)
        b16 = gr.Button("Cargar modelo", visible=False)
        
    mk16 = gr.Markdown(value="---", visible=False)
    b16.click(fn=load_downloaded_model, inputs=l16_url, outputs=results)
        
    with gr.Row(visible=False) as row17:
        l17 = gr.Markdown(value="", visible=False)
        l17_url = gr.Textbox("Label 1", visible=False)
        b17 = gr.Button("Cargar modelo", visible=False)
    
    mk17 = gr.Markdown(value="---", visible=False)    
    b17.click(fn=load_downloaded_model, inputs=l17_url, outputs=results)
        
    with gr.Row(visible=False) as row18:
        l18 = gr.Markdown(value="", visible=False)
        l18_url = gr.Textbox("Label 1", visible=False)
        b18 = gr.Button("Cargar modelo", visible=False)
        
    mk18 = gr.Markdown(value="---", visible=False)
    b18.click(fn=load_downloaded_model, inputs=l18_url, outputs=results)
        
    with gr.Row(visible=False) as row19:
        l19 = gr.Markdown(value="", visible=False)
        l19_url = gr.Textbox("Label 1", visible=False)
        b19 = gr.Button("Cargar modelo", visible=False)
        
    mk19 = gr.Markdown(value="---", visible=False)
    b19.click(fn=load_downloaded_model, inputs=l19_url, outputs=results)
        
    with gr.Row(visible=False) as row20:
        l20 = gr.Markdown(value="", visible=False)
        l20_url = gr.Textbox("Label 1", visible=False)
        b20 = gr.Button("Cargar modelo", visible=False)
    
    mk20 = gr.Markdown(value="---", visible=False)
    b20.click(fn=load_downloaded_model, inputs=l20_url, outputs=results)
    
    #   to_return_protect1 = 

    search_model_button.click(fn=get_models_by_name, inputs=model_name, outputs=[l1,l1_url, b1, mk1, row1,
                                                                                 l2,l2_url, b2, mk2, row2,
                                                                                 l3,l3_url, b3, mk3, row3,
                                                                                 l4,l4_url, b4, mk4, row4,
                                                                                 l5,l5_url, b5, mk5, row5,
                                                                                 l6,l6_url, b6, mk6, row6,
                                                                                 l7,l7_url, b7, mk7, row7,
                                                                                 l8,l8_url, b8, mk8, row8,
                                                                                 l9,l9_url, b9, mk9, row9,
                                                                                 l10,l10_url, b10, mk10, row10,
                                                                                 l11,l11_url, b11, mk11, row11,
                                                                                 l12,l12_url, b12, mk12, row12,
                                                                                 l13,l13_url, b13, mk13, row13,
                                                                                 l14,l14_url, b14, mk14, row14,
                                                                                 l15,l15_url, b15, mk15, row15,
                                                                                 l16,l16_url, b16, mk16, row16,
                                                                                 l17,l17_url, b17, mk17, row17,
                                                                                 l18,l18_url, b18, mk18, row18,
                                                                                 l19,l19_url, b19, mk19, row19,
                                                                                 l20,l20_url, b20, mk20, row20,
                                                                                 results
                                                                                 ])
def publish_model_clicked(name, url):
    
    ws_url = "https://script.google.com/macros/s/AKfycbzCBVubj9vkjKC8C0h-9-JnxUT4guKG2cs5dvDbnV2rSwNzzJ-WHvQab1WOvH0AAHowYg/exec"
    
    response = requests.post(ws_url, json={
        'type': 'model_by_url',
        'url': url
    })

    response_json = response.json()
    print(response_json)
    infos = []
    
    if len(name) < 10:
        infos.append("El nombre del modelo debe ser más descriptivo.")
        yield "\n".join(infos)
        
        return
    
    if response_json.get('exists') == True:
        infos.append("El enlace que intentas usar ya se encuentra en el sistema.")
        yield "\n".join(infos)
        
        return 
    
    zips_path = "zips"
    
    if os.path.exists(zips_path):
        shutil.rmtree(zips_path)
    
    os.mkdir(zips_path)
         
    download_file = download_from_url(url)
    
    dowloaded = False
    if not download_file:
        print("No se ha podido descargar el modelo.")
        infos.append("No se ha podido descargar el modelo.")
    elif download_file == "downloaded":
        print("Comprobando archivos del modelo...")
        infos.append("Comprobando archivos del modelo....")
        
        for filename in os.listdir(zips_path):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join(zips_path,filename)
                
                with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
                    contiene_index = False
                    contiene_pth = False
                    contiene_otro = False
                    
                    for archivo in zip_ref.namelist():
                        if archivo.endswith('.index'):
                            contiene_index = True
                        elif archivo.endswith('.pth') and not archivo.startswith("D_") and  not archivo.startswith("G_"):
                            contiene_pth = True
                        if (not archivo.endswith('.index') and not archivo.endswith("/")) and (archivo.endswith('.pth') and archivo.startswith("D_") and not archivo.endswith("/")) or (archivo.startswith("G_") and archivo.endswith('.pth') and not archivo.endswith("/")):
                            contiene_otro = True
                            
                    if contiene_index and contiene_pth and not contiene_otro:
                        response1 = requests.post(ws_url, json={
                            'type': 'save_model',
                            'url': url,
                            'name': name
                        })
                        infos.append("Modelo aceptado.")
                        print("Modelo aceptado.")
                    else:
                        infos.append("Modelo no aceptado.")
                        print("Modelo no aceptado.")
                    
                yield "\n".join(infos)
            else:
                print("Error al descomprimir el modelo.")
                infos.append("Error al descomprimir el modelo.")
                yield "\n".join(infos)    
    elif download_file == "demasiado uso":
        infos.append("El enlace llegó al limite de uso, intenta nuevamente más tarde o usa otro enlace.")
    
    shutil.rmtree(zips_path)
    yield "\n".join(infos)

def publish_models():
    with gr.Column():
        gr.Markdown("# Publicar un modelo en la comunidad")
        gr.Markdown("El modelo se va a verificar antes de publicarse. Importante que solo contenga el archivo **.pth** del modelo y el archivo **added_.index** para que no sea rechazado.")
        
        model_name = gr.inputs.Textbox(lines=1, label="Nombre descriptivo del modelo Ej: (Ben 10 [Latino] - RVC V2 - 250 Epoch)")
        url = gr.inputs.Textbox(lines=1, label="Enlace del modelo")
        publish_model_button=gr.Button("Publicar modelo")
        results = gr.Textbox(label="Resultado", value="", max_lines=20)
        
        publish_model_button.click(fn=publish_model_clicked, inputs=[model_name, url], outputs=results)
        
