# Article 1 de la série sur l'industrialisation des modèles de Machine Learning dans le cloud Azure

## Industrialisation d'un algorithme de Machine Learning en tant que service en temps réel.

<p align="justify">
L'industrialisation (ou le déploiement) d'un modèle de Mahcine Learning est la dernière étape du processus de construction d'un algorithme avant d'être utilisé en production. 
L'usage du cloud permet de faciliter cette étape complexe.
Il existe principalement deux types d'industrialisation : l'inférence en temps réel ou l'inférence par lot. 
Nous verrons dans cet article un exemple d'industrialisation en temps réel en utilisant le service Azure Machine Learning.
On industrialise un modèle prédisant si un client veut quitter la banque dans les prochains mois. Cet algorithme est présenté dans <a href="https://github.com/Teywa-OZIOL/Article_Data_Science_Katalyse_IS/blob/main/Articles/Serie_1_Article_3_Implementation_XGBoost_Python">cet article</a>. 
</p>

### Enregistrement des modèles

<p align="justify">
On enregistre le pipeline de preprocessing ainsi que le modèle en utilisant la fonction "dump()" des packages "joblib" et "pickle" sous python. Ces modèles sont construits en local.
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

### Inscription des modèles dans l'espace de travail

<p align="justify">
On commence par définir son espace de travail qui sera contenu dans la variable "ws". Il suffit d'utiliser la fonction "from_config()" qui lit un fichier contenant les informations requises pour faire la connexion avec l'espace de travail.
</p>

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```

<p align="justify">
On inscrit en suite les deux fichiers en tant que modèle dans l'espace de travail Azure Machine Learning. On précise le path de chacun des modèle ainsi que le nom que l'on a donné aux modèles. On indique l'espace de travail dans lequel on souhaite enregistrer les modèles.
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

### Création du script d'inférence

<p align="justify">
On crée un nouveau fichier python contenant le script d'inférence. On note que "script_file" désigne le chemin vers lequel le fichier sera enregistré. Un script d'inférence contient obligatoirement les fonctions "init()" et "run()". On peut ajouter d'autres fonctions que l'on pourrait avoir besoin pour transformer les données. La fonction "init()" permet d'initialiser nos deux modèles. Il faut récupérer le path de chacun des modèles à partir de leurs noms puis de les charger à l'aide de la fonction "load()" des packages "joblib" ou "pickle". On applique dans la fonction "run()" le modèle de preprocessing puis le modèle XGBoost aux nouvelles données. On retourne les prédictions des nouvelles données au format JSON. 
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

### Définition de l'envrionnement d'inférence

<p align="justify">
Il faut maintenant définir l'environnement python. Il faut ajouter certains packages en plus des packages de base pour que le script d'inférence puisse réaliser les calculs. On peut cloner l'environnement que l'on a utilisé pour le développement afin d'être certain que tous les packages soient disponibles. On peut aussi ajouter les packages souhaités un à un.
</p>

```python
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package('scikit-learn=0.22.1')
myenv.add_conda_package('xgboost')

env_file = os.path.join(folder_name,"scoring_env.yml")
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
```

### Définition de la configuration d'inférence et de déploiement puis déploiement du modèle

<p align="justify">
Une fois le script d'inférence et l'environnement définis, on possède tous les éléments de la configuration de l'inférence. On utilise la fonction "InferenceConfig()" en précisant le runtime puis le script d'inférence et l'environnement. Dans un second temps, on indique le conteneur que l'on souhaite ainsi que ces propriétés. Ici, j'utilise Azure Container Instance avec des performances minimales. Azure Container Instance est utilisé pour le développement et les tests. Pour de l'inférence en temps réel, il est préférable d'utiliser Azure Kubernetes Service en production. Les propriétés telles que le CPU et la mémoire dépendent de la fréquence et du volume des données à prédire, de la vitesse et du prix que l'on souhaite. On peut ensuite déployer le modèle. On précise l'espace de travail auquel ce déploiement est associé, le nom que l'on donne à ce service, les modèles et les configurations d'inférence et de déploiement définies plus tôt.
</p>

```python
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Model

inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "first-scoring-service"

service = Model.deploy(ws, service_name, [model1, model2], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)
```

<p align="justify">
On récupère l'uri désignant le conteneur ACI pour pourvoir interroger le modèle. L'industrialisation est maintenant terminée.
</p>

```python
endpoint = service.scoring_uri
print(endpoint)
```
<p align="justify">
On peut aussi supprimer le web service en le récupérant à partir de son nom puis en utilisant la fonction "delete()" d'AzureML.
</p>

```python
service = ws.webservices['first-scoring-service']
service.delete()
```
### Requetâge du modèle pour prédire si un individu souhaite quitter la banque

<p align="justify">
Maintenant que l'algorithme est déployé dans un conteneur hébergé dans le cloud Azure, on peut appeler l'algorithme pour scorer de nouvelles données. On quitte l'espace de travail et on ouvre un nouveau fichier indépendant des scripts précédents. On va envoyer une requête vers le service ACI qui héberge le modèle. Il faut plusieurs éléments pour que la requête puisse être valide. Il faut l'uri que l'on a récupéré précédemment, des authentifiants pour accéder au modèle que l'on précise dans "headers" (il n'y en a pas besoin ici) et les nouvelles données à scorer. On poste la requête et on récupère les résultats que l'on affiche.
</p>

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

<p align="justify">
Voici les résultats :
</p>

<p align="center">
  <img width="400" height="30" src="/Pictures/Image15.png">
</p>

<p align="justify">
Cet Allemand de 32 ans ne souhaite pas quitter la banque puisqu'il appartient à la classe 0. La probabilité d'appartenance à cette classe est élevée puisqu'elle est de 97.9%. Il y a donc 2.1% de chances que cet individu souhaite quitter la banque dans les prochains mois.
</p>

<p align="justify">
Après avoir déployé le modèle dans un Azure Container Service, on peut l'appeler à n'importe quel moment pour scorer de nouvelles données et savoir si d'autres clients souhaitent quitter leur banque. On peut ajouter un objet datacollector dans le script d'inférence pour récupérer les nouvelles données scorées et ainsi suivre les performances du modèle au cours du temps. Azure Machine Learing permet de faire ce suivi qui est indispensable lorsqu'un modèle est mis en production.
</p>
