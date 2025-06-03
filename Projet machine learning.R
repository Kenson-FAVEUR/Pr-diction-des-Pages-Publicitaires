# Importer les bibliothèques nécessaires
library(caret)
library(glmnet)
library(rpart)
library(randomForest)

# Chargement des données
setwd("C:/Users/PC/OneDrive/Bureau/Génération et évaluation modèle")
pub.data=read.table("pub.data",header=T)
table(pub.data$Y)

# Convertir Y en facteur
pub.data$Y<-factor(pub.data$Y)

# Aperçu les données
str(pub.data)
head(pub.data)
names(pub.data)

# Vérifier si il y a des valeurs manquantes
sum(is.na(pub.data))

# Analyse statistiques
summary(pub.data)
summary(pub.data$Y)

# Séparation des données en entrainnement et test
set.seed(2025)
trainindex <- createDataPartition(pub.data$Y, p=0.8, list = FALSE)
trainData <- pub.data[trainindex, ] # 80% d'entrainnement
testData <- pub.data[-trainindex,]  # 20%  Test

# Variables explicatives et cible
x.train<- trainData[, -ncol(trainData)]
y.train <- trainData$Y
x.test <- testData[, -ncol(testData)]
y.test <- testData$Y

# Vérification de la dimension  train et test
dim(trainData)
dim(testData)

# Vérification de la proportion des données
prop.table(table(trainData$Y))
prop.table(table(testData$Y))

# Après la séparation des données, appliquons les différentes algorithmes.
# Régression logistique
# On ajuste un modèle logistique classique sur les données d'entraînement.
# Ce modèle suppose une relation linéaire entre les variables et la log-odds de la classe cible.
# Il est sensible à la colinéarité entre variables.
logit.model <- glm(Y~., data = trainData, family = "binomial")
summary(logit.model)
predict.logit <- predict(logit.model, newdata = testData, type = "response")
predict.class <- ifelse(predict.logit > 0.5, "pub", "nonpub")
confusionMatrix(factor(predict.class), y.test)

# Appliquons la régression logistique Lasso
# Le Lasso (Régression pénalisée) permet une sélection automatique de variables.
# On effectue une validation croisée pour choisir le paramètre de régularisation optimal (lambda).
# Le modèle est ensuite testé sur les données de test.
x.train.mat <- as.matrix(x.train)
x.test.mat <- as.matrix(x.test)
y.train.bin <- ifelse(y.train == "pub", 1, 0)

cv.lasso <- cv.glmnet(x.train.mat, y.train.bin, alpha = 1, family = "binomial")

lasso.model <- glmnet(x.train.mat, y.train.bin,
                      alpha = 1,
                      family = "binomial",
                      maxit = 200000,
                      standardize = TRUE)

# Prédictions avec le meilleur lambda
pred.lasso <- predict(cv.lasso, newx = x.test.mat, s = "lambda.min", type = "response")

# Transformer en classes
pred.class.lasso <- ifelse(pred.lasso > 0.5, "pub", "nonpub")

# S'assurer que c’est bien un facteur avec les bons niveaux
pred.class.lasso <- factor(pred.class.lasso, levels = levels(y.test))

# Matrice de confusion
confusionMatrix(pred.class.lasso, y.test)


## Arbre de décision CART
# On utilise la méthode rpart pour construire un arbre de décision.
# Cette méthode est simple, interprétable, mais sensible à la variance.
library(rpart)
model.cart <- rpart(Y~., data = trainData, method = "class")
pred.cart <- predict(model.cart, testData, type ="class")
confusionMatrix(pred.cart, y.test)

# Appliquons le foret aléatoire
# La Random Forest agrège plusieurs arbres de décision construits sur des sous-échantillons.
# Elle est robuste, stable, et gère bien les données bruitées ou nombreuses variables.
# Le nombre d'arbres est fixé à 100 ; on peut ajuster ce paramètre pour voir la convergence.
library(randomForest)
model.rf <- randomForest(Y~., data = trainData, ntree = 100)
plot(model.rf)
pred.rf <- predict(model.rf, newdata = testData)
confusionMatrix(pred.rf, y.test)

# Comparaison des performances
# On calcule l'erreur de classification pour chaque modèle sur l’échantillon test.
# Ceci permet une comparaison objective des performances.
err.logit <- mean(predict.class != y.test)
err.lasso <- mean(pred.class.lasso != y.test)
err.cart <- mean(pred.cart != y.test)
err.rf <- mean(pred.rf != y.test)

data.frame(
  Méthode = c("Régression Logistique", "Lasso", "Cart", "Foret Aléatoire"),
  Erreur_Classification = c(err.logit, err.lasso, err.cart, err.rf)
)


## Prédiction pour une page web dans l'échantillon test
# On sélectionne une observation de l’échantillon test pour tester la capacité de prédiction individuelle.
# Ce test permet de voir quels modèles détectent correctement une page pub dans un cas concret.
# Sélection de la première page
sample.page <- testData[1, -ncol(testData)]
true.label <- testData$Y[1]

length(sample.page)
length(true.label)

# Prédiction sur les 4 modèles
pred.logit.ex <- ifelse(predict(logit.model, sample.page, type = "response") > 0.5, "pub", "nonpub")
pred.lasso.ex <- ifelse(predict(lasso.model, as.matrix(sample.page), type = "response") > 0.5, "pub", "nonpub")
pred.cart.ex <- predict(model.cart, sample.page, type = "class")
pred.rf.ex <- predict(model.rf, sample.page)

# Résumé des résultats
cat("Vraie classe :", true.label, "\n")
cat("Régression logistique :", pred.logit.ex, "\n")
cat("Régression LASSO :", pred.lasso.ex, "\n")
cat("CART :", pred.cart.ex, "\n")
cat("Forêt aléatoire :", pred.rf.ex, "\n")

levels(trainData$Y)


