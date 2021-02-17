# Article 4 de la série sur l'ensemble learning
## Implémentation d’XGBoost en python pour prédire les individus souhaitant quitter une banque (churn)

<p align="justify">
Cet article présente un exemple d’algorithme de Machine Learning utilisant XGBoost. L’objectif est de prédire quels sont les individus qui souhaitent quitter une banque dans les prochains mois.
</p>

### Le Voting / L’averaging 

<p align="justify">
Le voting est un modèle de classification qui consiste à entrainer plusieurs algorithmes de Machine Learning différents sur un même jeu de données.  Pour connaitre la prédiction finale, on effectue un vote entre l’ensemble des algorithmes entrainés. On peut utiliser deux types de vote pour définir la prédiction finale : le hard vote et le soft vote. Pour le hard vote, on compte le nombre d’algorithme prédisant chacune des classes. Par exemple, on a quatre algorithmes, les trois premiers algorithmes prédisent la classe 0 et le dernier algorithme prédit la classe 1. La prédiction finale sera ainsi la classe 0 car elle a obtenu la majorité lors du vote. On peut aussi affecté un poids à chaque modèle et faire un vote pondéré par ces poids pour connaitre la prédiction finale. Le soft vote consiste à faire la moyenne des probabilités d’appartenance à chaque classe. La prédiction finale sera la classe ayant la moyenne d’appartenance la plus élevée. On peut également pondérer ces moyennes par des poids. 
</p>  

<p align="justify">
L’averaging est utilisé pour une régression. C’est le même principe que le voting. On effectue la moyenne des prédictions de chaque algorithme pour connaitre la valeur finale prédite. Cette moyenne peut aussi être pondérée par des poids affectés à chacun des modèles. 
</p>  

### Le Stacking

<p align="justify">
Le stacking est une nouvelle technique d’ensemble learning. Comme pour le voting, on va entrainer plusieurs modèles de machine learning sur un même dataset. Chaque algorithme effectuera une prédiction sur chacun des individus. Cela permettra de construire un nouveau jeu de données contenant N lignes et M colonnes avec N le nombre d’individus et M le nombre d’algorithmes. On entrainera un dernier algorithme de machine learning sur ce nouveau jeu de données. Ce dernier algorithme ne s’entrainera donc pas sur les données initiales mais sur les prédictions de ces données réalisées par les autres algorithmes. Ce dernier algorithme nous donnera la prédiction finale d’un modèle de stacking.
</p>

<p align="justify">
De manière plus générale, le stacking consiste à empiler des couches de modèles (apprenants faibles) pour former un unique modèle (apprenant fort). On a une première couche de modèle qui effectue des prédictions pour chaque individu du dataset. On crée ensuite un nouveau jeu de données de taille n*m avec n le nombre d’individu de la base et m le nombre de modèle de la couche précédente. Les modèles de la seconde couche s’entraineront à partir des prédictions des modèles de la couche précédente. On peut définir autant de couches que l’on souhaite. Le modèle final prend en entrée les prédictions de chaque modèle de la couche précédente et propose en sortie une prédiction unique pour chaque individu. C’est le méta-modèle (ou meta-learner). En pratique, on a souvent une seule couche d’algorithmes puis le méta-modèle.
</p>

<p align="justify">
On peut faire une analogie entre le stacking et les réseaux de neurones. En effet, dans les deux cas, on a un jeu de données en entrée puis des couches successives avant d’avoir un objet de sortie effectuant la prédiction. Dans le stacking, les éléments de chaque couche sont des modèles de machine learning et l’objet de sortie est le méta-modèle. Dans un réseau de neurones, chaque élément des couches sont des neurones puis on a une couche de neurones finales correspondant au méta-modèle pour le stacking.
</p>

### Mise en place des algorithmes sous Python :

