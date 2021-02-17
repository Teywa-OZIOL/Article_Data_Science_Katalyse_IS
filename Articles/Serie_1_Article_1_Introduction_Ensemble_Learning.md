# Article 1 de la série sur l'ensemble learning
## Présentation de l’ensemble learning

<p align="justify">
Si vous disposez de données structurées et si vous souhaitez que votre algorithme de Machine Learning soit le plus performant possible, l'utilisation d'une méthode ensembliste s'avère souvent être le bon choix. D'ailleurs, l'algorithme roi des compétitions de Machine Learning (XGBoost) est un algorithme ensembliste.
</p>

<p align="justify">
 Nous définirons le concept d’ensemble learning dans ce premier article de la série puis nous expliquerons deux types d’ensemble learning : boosting et le bagging.
</p>

<p align="justify">
L'ensemble learning est une technique de machine learning qui consiste à entraîner plusieurs modèles (parfois des milliers) sur nos données puis de les agréger pour former un modèle final. Les modèles initiaux s'appellent les apprenants faibles (weak learner en anglais) et le modèle final est l'apprenant fort (strong learner).
</p>

<p align="justify">
Les apprenants faibles peuvent être toutes sortes d'algorithmes de machine learning (arbre de décision, SVM, régression logistique, …). L'idée de l'ensemble learning est qu’un grand nombre de modèles sera plus performant qu'un seul modèle. En effet, c'est en comparant les résultats de chacun des modèles que l'apprenant fort fournira une prédiction de meilleure qualité qu’un algorithme seul.
</p>

<p align="justify">
Dans des algorithmes d’ensemble learning, les apprenants faibles doivent respecter deux conditions. Ils doivent avoir une performance strictement supérieure à 50% et leurs résultats doivent être différents les uns des autres. En effet, si chacun des apprenants faibles a une performance inférieure à 50% alors la performance de l'apprenant fort tendra vers 0. Dans le cas contraire, la performance de l'apprenant fort tendra vers 1. La deuxième affirmation s'explique par le fait que si tous les apprenants faibles réalisent les mêmes bonnes prédictions et les mêmes erreurs alors l'apprenant fort n'aura pas de performances supérieures à chacun des apprenants faibles.
</p>

<p align="justify">
On distingue plusieurs types de techniques d'ensemble learning comme le boosting et le bagging. Il existe aussi le voting ou encore le stacking. FAIRE LIEN ARTICLE 4
</p>

<p align="justify">
Le bagging
</p>

<p align="justify">
Le bagging (ou bootstrap aggreging) consiste à entraîner chaque apprenant faible sur un échantillon des données. Cet échantillon de données est tiré au hasard, selon la méthode d'échantillonnage bootstrapp. Cette technique permet de respecter la condition de diversité des apprenants faibles puisque chaque modèle de base ne sera pas entraîné sur les mêmes données. Les modèles sont entraînés de manière parallèle. On regroupe ensuite les résultats de chaque apprenant faible pour former l'apprenant fort. Pour une tâche de classification, chaque apprenant faible prédit une classe et la prédiction finale est la classe prédite le plus grand nombre de fois. Pour une tâche de régression, la prédiction finale est la moyenne des prédictions de chaque arbre. 
</p>

<p align="justify">
Le boosting
</p>

<p align="justify">
Le boosting fonctionne d'une manière différente. Au lieu d'entraîner les apprenants faibles sur une partie aléatoire du dataset, les apprenants faibles sont entrainés sur l’ensemble du dataset. Chaque apprenant faible s'entraîne en prenant en compte les erreurs effectuées par l'apprenant faible précédent. Cela permet d'avoir de la diversification dans le résultat de chaque modèle. On dit que les modèles sont construits séquentiellement.
Si vous voulez en savoir plus sur la théorie concernant le boosting, </p>
[consultez cet article]() 

<p align="justify">
Pour l'implémentation sous python d'un algorithme de boosting, </p>  [cliquez ici](https://github.com/Teywa-OZIOL/Article_Data_Science_Katalyse_IS/blob/main/Articles/Serie_1_Article_1_Introduction_Ensemble_Learning.md)



[consultez cet article](https://github.com/Teywa-OZIOL/Article_Data_Science_Katalyse_IS/blob/main/Articles/Serie_1_Article_2_Explication_Mathematique_Boosting.md)
