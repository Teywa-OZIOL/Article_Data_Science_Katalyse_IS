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

<p align="center">
  <img width="700" height="700" src="/Pictures/Image14.png">
</p>

