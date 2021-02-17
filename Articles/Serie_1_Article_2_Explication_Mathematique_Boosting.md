# Article 2 de la série sur l'ensemble learning
## Explication mathématique des algorithmes de boosting 

<p align="justify">
Le boosting est un type d’algorithme de machine learning particulièrement performant sur des jeux de données structurées. Ce type d’algorithme appartient à la famille de l’ensemble learning. Si vous souhaitez savoir ce qu’est l’ensemble learning ainsi que les principes de bases du boosting, <a href="">lisez cet article</a>.
</p>

<p align="justify">
Nous présenterons ici les trois algorithmes de boosting les plus populaires. Nous parlerons d’abord d’Adaboost et du Gradient Boosting avant de s’intéresser à l’algorithme le plus prisé des compétitions de Machine Learning : l’Extreme Gradient Boosting.
</p>

<p align="justify">
Adaboost (ou adaptive boosting) est considéré comme le premier algorithme ensembliste de type boosting. Il a été inventé par Yoav Freund et Robert Schapire au début du siècle. Comme tout algorithme ensembliste, Adaboost combine des apprenants faibles (weak learner) pour former un apprenant fort. Les apprenants faibles sont des arbres de décision, plus précisément des stumps (Arbre de décision avec une racine et deux feuilles). Cette restriction sur la profondeur de l’arbre permet à l’algorithme Adaboost de gérer l’overfitting. 
Chacun des arbres de décision est construit de manière séquentielle en prenant en compte les erreurs de l’arbre précédent. Pour cela, un poids est initialisé pour chaque individu. A chaque itération, on teste plusieurs stumps et on garde le meilleur au sens du gini score. 
Pour ce stump, on calcule l’erreur totale. Pour une tâche de classification l’erreur de l’arbre j est la somme des poids des individus ayant été mal classé. Voici la formule :
</p>

<p align="justify">
On attribue ensuite un poids à cet arbre. Ce poids dépend de l’erreur totale de l’arbre. On note que espilon est un terme correctif pour éviter des opérations interdites.
</p>

<p align="justify">
Ensuite, on recalcule les poids des individus. Un individu ayant un poids élevé signifie que l’arbre de décision n’a pas réussi à correctement le prédire. Voici les formules pour un individu mal classé puis pour un individu bien classé :
</p>

<p align="justify">
On répète ce processus autant de fois que l’on souhaite d’arbres. Le fait d’affecter un poids au individu force l’arbre de décision suivant à s’entrainer en particulier sur ces individus. A la fin du processus itératif, nous avons un certain nombre d’arbres et un poids affecté pour chaque arbre. Les meilleurs arbres sont ceux ayant un poids important, ils auront plus de pouvoir dans la prise de décision dans cet algorithme. 
</p>

<p align="justify">
Si nous avons de nouvelles données, il suffit de les passer dans chacun des stumps. Chaque stump retourne une prédiction. Pour la classification, la prédiction final est le vote de chaque stump pondéré par leur poids. Si le vote pondéré pour la classification à 0 est supérieur au vote pondéré pour la classification à 1 alors on prédit à 0 la valeur de la nouvelle donnée. Pour la régression, la valeur finale sera la moyenne des résultats de chaque arbre pondéré par le poids de ce dernier. 
</p>

<p align="justify">
Le gradient boosting est une deuxième implémentation de boosting. Il reprend les principes de base d’Adaboost en apportant quelques corrections. En effet, les deux algorithmes initialisent un apprenant fort et créent de manière séquentielle des apprenants faibles qui s’ajoutent à l’apprenant fort. Ils diffèrent sur la façon dont ils créent les apprenants faibles au cours du processus itératif. À chaque itération, Adaboost modifie les poids attachés à chacun des individus. L’apprenant faible se concentre donc davantage sur les cas difficiles en prenant particulièrement en compte les individus ayant un poids élevé. Quant à lui, le Gradient Boosting ne modifie pas des poids pour chacun des individus. Au lieu de s’entraîner sur les données, l’apprenant faible s’entraîne à prédire les résidus. C’est-à-dire l’erreur entre la valeur réelle et la valeur prédite. Plus la valeur du résidu de l’individu est grande et moins les arbres précédents sont arrivés à prédire correctement l’individu. Une prédiction de base est effectuée pour pouvoir calculer les résidus à la première itération. Une fonction de coût est utilisée pour calculer les résidus. C’est une autre façon de donner plus d’importance aux cas difficiles. Voici la formule pour calculer les résidus à l’itération j. 
</p>


<p align="justify">
Un arbre de décision est ensuite créé pour prédire ces résidus. Pour chacun des arbres, il faut calculer la valeur de sortie de chacune de ses feuilles. La fonction de coût intervient une nouvelle fois. La valeur de sortie de la feuille k à l’itération j est : 
</p>

<p align="justify">
On note que T est le nombre de feuilles de l’arbre de décision.
Enfin, on ajoute l’arbre à l’apprenant fort pondéré par un learning rate. Le learning rate apparaît ici pour éviter de trop vite réduire les résidus dès les premières itérations pour éviter l’overfitting. C’est le choix de la fonction de coût qui permet de faire de la régression ou de la classification.
</p>

<p align="justify">
Prenons par exemple la fonction de coût suivante pour la régression :
</p>

<p align="justify">
On commence par initialiser l’apprenant fort F0(x) par une prédiction initiale. Pour la fonction de coût définie ci-dessus , la valeur de l’initialisation de chaque individu est la moyenne des valeurs de la variable cible. On calcule ensuite les résidus. Pour cette même fonction de coût, les résidus à l’itération j sont la différence entre la variable cible et F(j-1)(x) pour chaque individu. On entraine ensuite un arbre de décision sur ces résidus puis on calcule la valeur de sortie pour chaque feuille. Si l’on reprend la même fonction de coût, alors la valeur de sortie est la moyenne des résidus de la feuille. 
L’Extreme Gradient Boosting reprend les bases du Gradient Boosting, on initialise un apprenant fort en lui donnant une valeur de base, on entraine chaque arbre pour prédire les résidus et il faut minimiser une fonction de coût. Cependant, plusieurs points diffèrent :
</p>

<p align="justify">
- Nous avons la fonction suivante que l’on cherche à minimiser
- L’entraînement de l’arbre est plus complexe. On va calculer un score de similarité puis un score de gain pour s’assurer que le meilleur arbre possible est construit. Voici la formule du score de similarité puis du score de gain. On répète cela pour construire un arbre ayant une profondeur prédéfinie. La profondeur de l’arbre est un hyperparamètre de l’algorithme. 
- Une fois que l’arbre est construit, on va l’élaguer pour enlever les branches inutiles. Il n’y avait pas d’élagage pour le Gradient Boosting. Il faut fixer une valeur d’élagage, ce qui constitue un nouvel hyperparamètre de l’algorithme (gamma dans XGBoost).
- On finit par calculer les valeurs de sortie pour chaque feuille de l’arbre :
- On gère l’overfitting de trois manières différentes (dans le score de similarité, dans le calcul des valeurs de sorties et dans l’ajout de l’arbre à l’apprenant fort en ajoutant un learning rate). Pour le Gradient Boosting, on a seulement le learning rate pour gérer l’overfitting lorsque l’on ajoute l’arbre à l’apprenant fort. L’hyperparamètre qui intervient dans le score de similarité et dans le calcul des valeurs de sorties est lambda.
</p>

<p align="justify">
Cet algorithme est extrêmement performant dans les compétitions de Machine Learning. Si vous souhaitez voir l’implémentation de XGBoost sur python, cliquez ici.
</p>
