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


<p align="center">
  <img width="700" height="700" src="/Pictures/Image14.png">
</p>

```python
y_pred = xgb2.predict(X_test)
```
