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
