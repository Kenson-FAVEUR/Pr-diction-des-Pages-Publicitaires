# Prédiction-des-Pages-Publicitaires
# 📊 Prédiction des Pages Publicitaires

## 🎯 Objectif

Développer des modèles de machine learning capables de **classer automatiquement** des pages web comme **publicitaires (pub)** ou **non-publicitaires (nonpub)** à partir de **1554 variables numériques** décrivant leur contenu.

##  Données

- **Total observations** : 3279  
- **Variables explicatives** : 1554  
- **Variable cible** : `Y` (`pub` / `nonpub`)  
- **Partition** :
  - 80% pour l'entraînement (2624 observations)
  - 20% pour le test (655 observations)
- **Aucune valeur manquante**
- **Classes déséquilibrées** : `nonpub ≈ 86%`, `pub ≈ 14%`

##  Méthodes Utilisées

- Régression Logistique
- Régression Lasso (GLM pénalisé)
- Arbre de Décision (CART)
- Forêt Aléatoire (Random Forest)

## 📈 Résultats

| Modèle                  | Erreur (%) | Sensibilité | Spécificité | Balanced Accuracy |
|-------------------------|------------|-------------|-------------|-------------------|
| Régression Logistique   | 8.4        | 94.5%       | 73.6%       | 84.1%             |
| Régression Lasso        | 3.36       | 98.8%       | 83.5%       | 91.1%             |
| Arbre de Décision (CART)| 4.42       | 98.6%       | 76.9%       | 87.8%             |
| Forêt Aléatoire         | 3.05       | 99.1%       | 83.5%       | 91.3%             |

 **Meilleure performance globale** obtenue avec la Forêt Aléatoire.



