# Article 3 de la série sur l'ensemble learning
## Implémentation d’XGBoost en python pour prédire les individus souhaitant quitter une banque (churn)

<p align="justify">
Cet article présente un exemple d’algorithme de Machine Learning utilisant XGBoost. L’objectif est de prédire quels sont les individus qui souhaitent quitter une banque dans les prochains mois.
</p>

<p align="justify">
Comme nous l’avons vu dans un articles précédent, XGBoost est un algorithme de boosting qui est très performant sur des données structurées. Il possède un nombre d’hyperparamètres important permattant d’avoir une maitrise totale de l’entrainement du modèle et notamment un contrôle du sur-apprentissage.
</p>  

<p align="justify">
Le dataset a été téléchargé sur le site kaggle disponible en cliquant sur : <a href="https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers">ce lien</a>
</p>  

## Présentation de l’algorithme :

### Analyse exploratoire

<p align="justify">
Après avoir chargé les données dans un notebook jupyter, on réalise une analyse exploratoire afin de bien les comprendre. Nous avons une dizaine de variables explicatives à notre disposition pour élaborer l'algorithme prédictif. Ces variables nous apportent de l'information sur l'individu. Nous avons ainsi des variables comme l’âge de l’individu, son sexe ou encore la localisation de l’individu. Il y a aussi le nombre d’année de l’individu en tant que client dans la banque, s’il possède une carte de crédit et s’il est un membre actif pour la banque. Nous avons enfin le score de crédit attribué à l’individu, sa balance sur ses comptes, le nombre de produits financiers auquel il a souscrit et l’estimation de son salaire. Les variables “RowNumer” , “CustomerId” et “Surname” ont été supprimé car elles sont inutiles. La variable cible est la variable binaire « Exited » (0 si le client veut rester et 1 s’il veut partir). 
</p>

<p align="justify">
La phase d’analyse exploratoire nous apprend qu’il n’y a pas de valeurs manquantes. Les variables « Geography » et « Gender » sont des variables qualitatives sous la forme de chaines de caractères. Les autres variables possèdent des valeurs numériques. Les variables « HasCrCard » et « IsActiveMember » sont qualitatives et les autres sont quantitatives. On analyse la distribution de ces variables. On remarque par exemple que les personnes ont soit une balance nulle soit une balance de plus de 50 000 euros. Les autres variables suivent une distribution gaussienne. Les clients possèdent souvent un ou deux produits financiers. Il y a uniquement de faibles corrélations entre les variables. On s’aperçoit que les personnes souhaitant quitter la banque sont plutôt des femmes, des membres inactifs et des allemands. Les variables âge et balance jouent également un rôle clé puisque les personnes qui souhaitent quitter la banque sont plus âgés que les personnes souhaitant y rester. Les personnes qui possèdent une balance nulle ne souhaite pas quitter la banque dans un cas général. Voici un extrait des graphiques permettant de décrire les éléments précédents : </p>

METTRE GRAPHIQUE

<p align="justify">
Il faut prendre en compte l'ensemble de ces informations lors de l'élaboration de la phase de preprocessing des données.
</p>

### Preprocessing

<p align="justify">
On peut maintenant réaliser le preprocessing des données. Nous créons au préalable une base d’entrainement et une base de test en utilisant la fonction « train_test_split() ». Voici le preprocessing effectué :
</p>

```python
def feature_engineering_and_discretization(df2):
    df = df2.copy()
    df["CreditScoreDiscr"] = pd.qcut(df["CreditScore"], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df["AgeScoreDiscr"] = pd.qcut(df["Age"], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df["BalanceScoreDiscr"] = pd.qcut(df["Balance"], 2, labels = [1, 2])
    df["EstimatedSalaryDiscr"] = pd.qcut(df["EstimatedSalary"], 10, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    df["Balance_parrapport_salaire"] = df["Balance"]/df["EstimatedSalary"]
    df["Tenure_par_age"] = df["Tenure"]/(df["Age"])
    df["CreditScore_par_Age"] = df["CreditScore"]/(df["Age"])
    df["Salaire_par_age"] = df["EstimatedSalary"]/(df["Age"])
    return df
transformer_fe = FunctionTransformer(feature_engineering_and_discretization)

def null_func(x):
    return x
transformer_null = FunctionTransformer(null_func)

Cat_feat = ['HasCrCard','IsActiveMember']
Num_feat = ['CreditScore','Age','Balance','EstimatedSalary','Tenure','NumOfProducts']
Ohe_feat = ['Geography','Gender']

ohe_pipe = Pipeline([('encoder', OneHotEncoder(drop='first', sparse=False))])

num_pipe = Pipeline([('scaler', StandardScaler())])

cat_pipe = Pipeline(steps=[('notransform', transformer_null)])

preprocessor = ColumnTransformer(transformers=[('ohe', ohe_pipe, Ohe_feat),
                                               ('num', num_pipe, Num_feat)])

pipe_preprocessing = Pipeline([('preprocessing', preprocessor)])

X_train = pipe_preprocessing.fit_transform(X_train,y_train)
X_test = pipe_preprocessing.transform(X_test)
```

<p align="justify">
On commence par une étape de feature engeneering pour créer de nouvelles variables à partir de celles que l’on a déjà. On crée notamment trois nouvelles variables en divisant une variable existante par l’âge. Par exemple, on divise la variable estimant le salaire de l’individu par son âge. Cette variable permettra à l’algorithme de comparer les salaires estimés par rapport à l’âge des individus . On effectue aussi une discrétisation des variables continues. Pour les variables « Geography » et « Gender », on réalise un encodage « one_hot » permettant de créer autant de variables qu’il y a de modalités différentes. On supprimer la première variable car elle n’apporte pas d’information supplémentaire. On standardise les variables numériques. L’ensemble de ces étapes sont rassemblés dans une pipeline finale qui est donc la pipeline de preprocessing. Cette pipeline permet d’industrialiser le modèle plus rapidement. On applique la pipeline aux données.
</p>

<p align="justify">
Avant d’entrainer le modèle, on réalise un sur-échantillonage pour rééquilibrer les classes car le nombre de personnes souhaitant quitter la banque est largement inférieur au nombre de personnes souhaitant rester.
</p>

```python
oversample = SVMSMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
```


### Entrainement du modèle

<p align="justify">
On peut maintenant entrainer le modèle. On construit une grille de recherche et on teste plusieurs combinaisons d’hyperparamètres afin de trouver le meilleur modèle possible. L’optimisation se fait au sens de l’AUC (aire sous la courbe ROC) car les classes sont déséquilibrés. 
</p>

```python
params = {
        'booster' : ['gbtree'],
        'eta' : [0,0.05,0.1,0.2,0.4],
        'min_child_weight': [1, 5, 10],
        'alpha': [0,0.05,0.5],
        'gamma': [0,0.01,0.05],
        'lambda' : [1,0,0.5],
        'max_depth': [3, 4, 5],
        'n_estimators' : [100,500,1000]
        }

xgb = XGBClassifier(objective='binary:logistic', nthread=1)

folds = 2
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', 
                                   n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state = 42)

random_search.fit(X_train, y_train)
```

<p align="justify">
Le modèle qui maximise l’AUC est le modèle suivant :
</p>

```python
xgb2 = XGBClassifier(alpha=0.5, base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eta=0.2, gamma=0,
              gpu_id=-1, importance_type='gain', interaction_constraints='',
              learning_rate=0.200000003, max_delta_step=0,
              max_depth=5, min_child_weight=10, missing=np.nan,
              monotone_constraints='()', n_estimators=100, n_jobs=1, nthread=1,
              num_parallel_tree=1, random_state=0, reg_alpha=0.5, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)

xgb2.fit(X_train,y_train)
```

### Analyse des résultats

<p align="justify">
Après l’entrainement de l’algorithme, on prédit les valeurs pour les individus de la base de test. Nous pouvons obtenir directement la classe auquel l’individu appartient ou la probabilité d’appartenance à cette classe. Etant donné que les deux classes de la variable cible sont déséquilibré, la métrique la plus pertinente pour l’analyse des performances du modèle est l’AUC. Nous obtenons un score 0.75 par rapport à cette métrique. On peut aussi imaginé une optimisation par rapport au rappel ou à la précision en fonction des demandes de la banque concerné.
</p>

<p align="justify">
Le rappel est la proportion de réels positifs que l’algorithme a correctement classé. La précision est la proportion d’individus classés comme positif et qui le sont vraiment. On peut modifier le seuil d’attribution des classes pour obtenir un rappel élevé mais avec une précision plus faible en contre partie ou avoir une forte précision mais avec un faible rappel. 
</p>
