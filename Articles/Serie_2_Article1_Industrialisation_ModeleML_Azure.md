# Article 1 de la série sur l'industrialisation des modèles de Machine Learning dans le cloud Azure

## Industrialisation d'un algorithme de Machine Learning en tant que service en temps réel.

<p align="justify">
L'industrialisation (ou le déploiement) d'un modèle de Mahcine Learning est la dernière étape du processus de construction d'un algorithme avant de pouvoir l'utiliser. 
L'utilisation du cloud permet de faciliter grandement cette étape qui peut-être complexe.
Il existe principalement deux types d'industrialisation : l'inférence en temps réel ou l'inférence par lot. 
Nous verrons dans cet article un exemple d'industrialisation en temps réel en utilisant le service Azure Machine Learning. 
</p>

<p align="justify">
On industrialise le modèle prédisant si un client veut quitter la banque dans les prochains mois. Cet algorithme est présenté dans <a href="https://github.com/Teywa-OZIOL/Article_Data_Science_Katalyse_IS/blob/main/Articles/Serie_1_Article_3_Implementation_XGBoost_Python">cet article</a>. On enregistre la pipeline de preprocessing ainsi que le modèle en utilisant la fonction "dump()" des packages "joblib" et "pickle" sous python. Ces modèles sont construits en local.
</p>

```python
from joblib import dump
import pickle

dump(pipe_preprocessing, 'pipeline_processing.pkl')
pickle.dump(xgb_model, open("modelML2.pkl", "wb"))
```

<p align="justify">
On dispose de ces deux fichiers puis on utilise un espace de travail sous Azure Machine Learning.
</p>

<p align="justify">
On commence par définir son espace de travail qui sera contenu dans la variable "ws".
</p>

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```

<p align="justify">
On inscrit en suite les deux fichiers en tant que modèle dans l'espace de travail Azure Machine Learning.
</p>

```python
from azureml.core.model import Model

model1 = Model.register(model_path="./pipeline_processing.pkl",
                    model_name="pipeline_processing",
                    workspace=ws)

model2 = Model.register(model_path="./modelML2.pkl",
                    model_name="modelML2",
                    workspace=ws)
```
<p align="justify">
On crée un nouveau fichier python contenant le script d'inférence. On note que "script_file" désigne le chemin vers lequel le fichier sera enregistré.
</p>

```python
%%writefile $script_file
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model_pipe
    global model_xgb
        
    model_path1 = Model.get_model_path('pipeline_processing')
    model_pipe = joblib.load(model_path1)
    model_path2 = Model.get_model_path('modelML2')
    model_xgb = pickle.load(open(model_path2, "rb"))


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    df = pd.DataFrame(data ,columns=["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"])  
    predictions = model_xgb.predict_proba(model_pipe.transform(df))
    return json.dumps(predictions.tolist())
```
```python
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package('scikit-learn=0.22.1')
myenv.add_conda_package('xgboost')

env_file = os.path.join(folder_name,"scoring_env.yml")
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())

<p align="center">
  <img width="700" height="700" src="/Pictures/Image14.png">
</p>
```

```python
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Model

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "first-scoring-service"

service = Model.deploy(ws, service_name, [model1, model2], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)
```
```python
endpoint = service.scoring_uri
print(endpoint)
```

```python
import requests
import json

endpoint = "http://8aaf3742-a34f-4abe-8473-be7eb300dda6.francecentral.azurecontainer.io/score"

x_new = [[596, "Germany", "Male", 32, 3, 96709.1, 2, 0, 0, 41788.4]]

input_json = json.dumps({"data": x_new})

headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
predicted_classes = json.loads(predictions.json())

print(predicted_classes)
```

```python
service = ws.webservices['first-scoring-service']
service.delete()
```

<p align="center">
  <img width="400" height="40" src="/Pictures/Image15.png">
</p>
