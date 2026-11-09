#################### Projet scoring - Iranian Churn #########################
# Etape 1 : description de la base

rm(list = ls())
library(corrplot)

# Lecture des donnees (mettre le fichier dans le dossier de travail)
data <- read.csv("Customer Churn.csv")
names(data) <- gsub("\\.+", ".", names(data))   # certains noms ont des doubles espaces

# Apercu de la base
dim(data)
str(data)
summary(data)

# Valeurs manquantes et doublons
colSums(is.na(data))
sum(duplicated(data))
data <- data[!duplicated(data), ]   # Suppression des doublons( une ligne = un client)
dim(data)

# Variable cible
data$Churn <- factor(data$Churn, levels = c(0, 1), labels = c("non", "oui")) # transforme churn en deux modalité
# (oui et non ) ; important pour le randomforest
table(data$Churn)
prop.table(table(data$Churn))
barplot(table(data$Churn), main = "Repartition du churn")

# Variables quantitatives selon le churn
vars <- c("Call.Failure", "Subscription.Length", "Seconds.of.Use", "Frequency.of.use",
          "Frequency.of.SMS", "Distinct.Called.Numbers", "Age", "Customer.Value")
aggregate(data[, vars], list(Churn = data$Churn), mean)

par(mfrow = c(2, 4))
for (v in vars) boxplot(data[[v]] ~ data$Churn, main = v, xlab = "", ylab = "")
par(mfrow = c(1, 1))

# Variables qualitatives : taux de churn par modalite


prop.table(table(data$Complains, data$Churn), 1)
prop.table(table(data$Tariff.Plan, data$Churn), 1)
prop.table(table(data$Status, data$Churn), 1)
prop.table(table(data$Charge.Amount, data$Churn), 1)
prop.table(table(data$Age.Group, data$Churn), 1)

table(data$Charge.Amount)
table(data$Age.Group)

# Correlations
corrplot(cor(data[, names(data) != "Churn"]), method = "number", type = "upper", number.cex = 0.6)



# Etape 2 : k-means et CAH 

library(cluster)

# Variables utilisees : pas la cible, et pas Age.Group (redondante avec Age)
X <- scale(data[, !(names(data) %in% c("Churn", "Age.Group", "Frequency.of.use", "Customer.Value"))])
 
#création d'une table où toutes les valeurs sont stadardisée par rapport à la moyenne avec scale()


####### kmeans #######

# Nombre de classes : methode du coude
max_clusters <- 10
wcss_values <- numeric(max_clusters)
for (k in 1:max_clusters) {
  wcss_values[k] <- kmeans(X, centers = k, nstart = 25)$tot.withinss
}
plot(1:max_clusters, wcss_values, type = "b", pch = 19,
     xlab = "Nombre de clusters", ylab = "WCSS")

# k-means avec le k choisi sur le coude
k <- 3
set.seed(123)
km.out <- kmeans(X, centers = k, nstart = 25)
km.out$size

# Qualite : R2 et pseudo F (vus en cours)
R2 <- km.out$betweenss / km.out$totss
pseudoF <- (R2 / (k - 1)) / ((1 - R2) / (nrow(X) - k))
R2
pseudoF

for (k in 2:5) {
  set.seed(123)
  km <- kmeans(X, centers = k, nstart = 25)
  R2 <- km$betweenss / km$totss
  pseudoF <- (R2 / (k - 1)) / ((1 - R2) / (nrow(X) - k))
  print(c(k = k, R2 = R2, pseudoF = pseudoF))
}
k <- 3



# Profil des classes
aggregate(data[, colnames(X)], list(Cluster = km.out$cluster), mean)
prop.table(table(km.out$cluster, data$Churn), 1)

####### CAH #######

d <- dist(X, method = "euclidean")

# Lien complet
hc1 <- hclust(d, method = "complete")
plot(hc1, labels = FALSE, hang = -1, main = "CAH - lien complet")

# Methode de Ward
hc2 <- hclust(d, method = "ward.D2")
plot(hc2, labels = FALSE, hang = -1, main = "CAH - Ward")
rect.hclust(hc2, k = k, border = 2:5)

# Hauteurs des dernieres fusions : couper apres le dernier grand saut
rev(hc2$height)[1:10]

# Decoupage en k classes et profil
sub_grp <- cutree(hc2, k = k)
table(sub_grp)
aggregate(data[, colnames(X)], list(Cluster = sub_grp), mean)
prop.table(table(sub_grp, data$Churn), 1)

# Comparaison CAH / k-means
table(CAH = sub_grp, Kmeans = km.out$cluster)

centres <- aggregate(X, list(sub_grp), mean)[, -1]
km.mixte <- kmeans(X, centers = centres)
km.mixte$size
km.mixte$betweenss / km.mixte$totss
prop.table(table(km.mixte$cluster, data$Churn), 1)




library(caret)
library(MASS)
library(klaR)
library(randomForest)

####### Separation train / test (70 % / 30 %) #######
set.seed(123)
indices <- createDataPartition(data$Churn, p = 0.7, list = FALSE)
train_data <- data[indices, ]
test_data  <- data[-indices, ]

prop.table(table(train_data$Churn))
prop.table(table(test_data$Churn))

# Variables candidates : on retire les 3 variables redondantes (correlation > 0,9)
vars <- setdiff(names(data), c("Churn", "Age.Group", "Frequency.of.use", "Customer.Value"))

###### Sélection de variables par LDA ###########

set.seed(123)
step_lda <- stepclass(x = train_data[, vars], grouping = train_data$Churn,
                      method = "lda", direction = "both",
                      improvement = 0.001, fold = 10, output = FALSE)

vars_finales <- as.character(step_lda$model$name)
vars_finales
# ne donne que Complains comme variable, or lui seul ne suffit pas pour constyruire les modèles

          ########### LDA Backward##########
set.seed(123)
st_bw <- stepclass(x = train_data[, vars], grouping = train_data$Churn,
                   method = "lda", direction = "backward",
                   improvement = 0.001, fold = 10)
st_bw

vars_finales <- as.character(st_bw$model$name)
vars_finales
# la meilleur méthode qui donne 8 variables 

          ########### LDA forward ##########
set.seed(123)
st_fw <- stepclass(x = train_data[, vars], grouping = train_data$Churn,
                   method = "lda", direction = "forward",
                   improvement = 0.001, fold = 10)
st_fw

vars_forward <- as.character(st_fw$model$name)
vars_forward
# ne donne que Complains comme variable, or lui seul ne suffit pas pour constyruire les modèles

