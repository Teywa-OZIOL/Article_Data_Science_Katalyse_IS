# Article 4 de la série sur l'ensemble learning
## Présentation du voting, de l'averaging et du stacking

<p align="justify">
Nous évoquerons dans cet article deux nouveaux concepts de l’ensemble learning : le voting (ou l’averaging) et le stacking. 
Si vous voulez en savoir plus sur les principes de l’ensemble learning, 
<a href="https://github.com/Teywa-OZIOL/Article_Data_Science_Katalyse_IS/blob/main/Articles/Serie_1_Article_1_Introduction_Ensemble_Learning.md">consultez cet article</a>.
</p>

### Le Voting / L’averaging 

<p align="justify">
Le voting est un modèle de classification qui consiste à entrainer plusieurs algorithmes de Machine Learning différents sur un même jeu de données. Pour connaître la prédiction finale, on effectue un vote entre l’ensemble des algorithmes entrainés. On peut utiliser deux types de vote pour définir la prédiction finale : le hard vote et le soft vote. Pour le hard vote, on compte le nombre d’algorithmes prédisant chacune des classes. Par exemple, on a quatre algorithmes, les trois premiers algorithmes prédisant la classe 0 et le dernier algorithme prédit la classe 1. La prédiction finale sera ainsi la classe 0 car elle a obtenu la majorité lors du vote. On peut aussi affecter un poids à chaque modèle et faire un vote pondéré par ces poids pour connaître la prédiction finale. Le soft vote consiste à faire la moyenne des probabilités d’appartenance à chaque classe. La prédiction finale sera la classe ayant la moyenne d’appartenance la plus élevée. On peut également pondérer ces moyennes par des poids. 
</p>  

<p align="justify">
L’averaging est utilisé pour une régression. C’est le même principe que le voting. On effectue la moyenne des prédictions de chaque algorithme pour connaître la valeur finale prédite. Cette moyenne peut aussi être pondérée par des poids affectés à chacun des modèles. 
</p>  

### Le Stacking

<p align="justify">
Le stacking est une nouvelle technique d’ensemble learning. Comme pour le voting, on va entrainer plusieurs modèles de machine learning sur un même dataset. Chaque algorithme effectuera une prédiction sur chacun des individus. Cela permettra de construire un nouveau jeu de données contenant N lignes et M colonnes avec N le nombre d’individus et M le nombre d’algorithmes. On entrainera un dernier algorithme de machine learning sur ce nouveau jeu de données. Ce dernier algorithme ne s’entrainera donc pas sur les données initiales mais sur les prédictions de ces données réalisées par les autres algorithmes. Ce dernier algorithme nous donnera la prédiction finale d’un modèle de stacking.
</p>

<p align="justify">
De manière plus générale, le stacking consiste à empiler des couches de modèles (apprenants faibles) pour former un unique modèle (apprenant fort). On a une première couche de modèle qui effectue des prédictions pour chaque individu du dataset. On crée ensuite un nouveau jeu de données de taille n*m avec n le nombre d’individus de la base et m le nombre de modèles de la couche précédente. Les modèles de la seconde couche s’entraineront à partir des prédictions des modèles de la couche précédente. On peut définir autant de couches que l’on souhaite. Le modèle final prend en entrée les prédictions de chaque modèle de la couche précédente et propose en sortie une prédiction unique pour chaque individu. C’est le méta-modèle (ou meta-learner). En pratique, on a souvent une seule couche d’algorithmes puis le méta-modèle.
</p>

<p align="justify">
On peut faire une analogie entre le stacking et les réseaux de neurones. En effet, dans les deux cas, on a un jeu de données en entrée puis des couches successives avant d’avoir un objet de sortie effectuant la prédiction. Dans le stacking, les éléments de chaque couche sont des modèles de machine learning et l’objet de sortie est le méta-modèle. Dans un réseau de neurones, chaque élément des couches sont des neurones puis on a une couche de neurones finales correspondant au méta-modèle pour le stacking.
</p>

### Mise en place des algorithmes sous Python

<p align="justify">
Il faut commercer par définir un certain nombre d'algorithmes. Nous souhaitons ici effectuer une classification, nous créons donc six classifiers. Nous avons un SVM, une forêt aléatoire, un KNN, une régression logistique, Adaboost et XGBoost. Ces classifiers sont optimisés par rapport aux données. On peut optimiser les algorithmes un après l'autre, mais le mieux est d'optimiser l'ensemble des classifiers en même temps (pour le voting ou le stacking). En effet, l'objectif n'est pas que chaque algorithme soit le plus performant possible mais c'est que l'ensemble des classifiers offre la performance optimale. Il est peut-être préférable d'avoir un classifier moins performant mais plus diversifié car lorsqu'il sera assemblé avec les autres, la performance totale sera potentiellement meilleure qu'avec des algorithmes optimisés un à un.
</p>

```python
classifier1 = SVC(C = 50, kernel = "rbf", probability = True)
classifier2 = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_depth = 10,
                                     max_features = "auto", n_jobs = 4, random_state = 1000)
classifier3 = KNeighborsClassifier(n_neighbors=25,algorithm = 'kd_tree',n_jobs=4) 
classifier4 = LogisticRegression(solver = 'newton-cg')
classifier5 = AdaBoostClassifier(n_estimators = 50, learning_rate = 1)
classifier6 = XGBClassifier(base_score=0.5, booster='gbtree', eta = 0.3, gamma=5,
              max_depth=5,n_estimators=100, reg_lambda=5)
````

##### Le voting

<p align="justify">
Pour définir l'algorithme de voting, on peut directement utiliser la fonction "VotingClassifier" de scikit learn. On précise dans cette fonction l'ensemble des classifiers que l'on souhaite utiliser. On indique aussi la stratégie que l'on choisit. Ici, j'utilise un vote de type "soft" en affectant des poids pour chaque algorithme. La forêt aléatoire, Adaboost et XGBoost ont un poids deux fois plus important que les autres algorithmes. Il suffit ensuite d'entrainer le modèle et d'effectuer des prédictions sur la base de test pour évaluer la performance de notre algorithme.
</p>

```python
clf = VotingClassifier(estimators=[('SVM', classifier1), ('RF', classifier2), 
                                     ('KNN', classifier3), ('LR', classifier4),
                                     ('AdB', classifier5), ('XGB', classifier6)], voting='soft',weights=[1,2,1,1,2,2])

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
````

##### Le stacking

<p align="justify">
Pour effectuer du stacking sous python, on utilise la fonction "StackingClassifier" de scikit-learn. On construit ici un modèle de stacking contenant une couche de modèle puis le méta-modèle final qui est une régression logistique. On précise donc l'ensemble des classifiers que l'on souhaite utiliser puis l'estimateur final (la régression logistique). On peut effectuer ces traitements de manière parallèle. On entraine l'algorithme final de stacking qui entrainera chacun des sous-modèles utilisés. On peut ensuite déterminer les prédictions sur la base de test pour évaluer les performances de la solution.
</p>

```python
estimators=[('RF', classifier2),('KNN', classifier3), ('LR', classifier4),
            ('AdB', classifier5),('XGB',classifier6)]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(solver = 'newton-cg'), verbose = 2, n_jobs = 4)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
````

##### Résultats des deux méthodes

<p align="justify">
Une fois l'entrainement des algorithmes terminés, on peut comparer les solutions de stacking ou de voting avec d'autres solutions et l'on peut répondre à plusieurs interrogations. On saura si l'utilisation du stacking ou du voting est pertinent. La performance totale de ces deux solutions sera peut-être inférieure à un algorithme de boosting ou de bagging entrainé de manière indépendante ou un autre algorithme non ensembliste de machine learning (SVM, Régression logistique, ...). Dans le cas contraire, l'utilisation de cette technique permettra peut-être d'améliorer de manière significative les performances de notre modèle.
</p>
