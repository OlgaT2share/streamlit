import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("train.csv")
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


#Explotation
if page == pages[0] : 
  st.write("### Introduction")
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())
  if st.checkbox("Afficher les NA") :
     st.dataframe(df.isna().sum())



#DATAVIZ
if page == pages[1] : 
  st.write("### DataVizualization")
  fig = plt.figure()
  sns.countplot(x = 'Survived', data = df, hue='Survived')
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Sex', data = df,hue='Sex')
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Pclass', data = df,hue='Pclass')
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)
  
  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)

  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)

  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig)

  fig, ax = plt.subplots()
  numeric_df = df.select_dtypes(include=['int64', 'float64'])
  sns.heatmap(numeric_df.corr(), ax=ax)
  st.write(fig)



#MODELISATION
if page == pages[2] : 
  st.write("### Modélisation")
  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
  for col in X_cat.columns:
       X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
       X_num[col] = X_num[col].fillna(X_num[col].median())

  #get_dummies :retourne pour chque valeur des valeur booleens, qui serons utile 
  # dans train_test_split
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  X = pd.concat([X_cat_scaled, X_num], axis = 1)

  #train_test_split permet de séparer les données en train et test,
  # Utilisez cette fonction pour éviter le surapprentissage et valider correctement vos modèles ! 
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  #StandardScaler est une méthode de normalisation (mise à l'échelle des données) très utilisée en Machine Learning 
  # pour standardiser les features (variables explicatives)
  #elle permet de centrer et reduire les doees pour quelles suivent une distribution centrée en 0 et d'écart-type 1
  #Utile pour les algorithmes sensibles aux échelles 
  #Évite qu'une feature avec des valeurs élevées domine le modèle((SVM, KNN, PCA, régression linéaire, réseaux de neurones...).)

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])# Calcule μ,σ ET standardise X_train
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])# Utilise μ et σ de X_train pour standardiser X_test
  
  #A travers cette fonction on peut A facilement comparer plusieurs algorithmes en changeant juste le nom du classifieur !
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix

  def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()  # Initialise une Forêt Aléatoire
    elif classifier == 'SVC':
        clf = SVC()                    # Initialise un SVM
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()     # Initialise une Régression Logistique
    
    clf.fit(X_train, y_train)          # Entraîne le modèle sur X_train, y_train
    return clf                         # Retourne le modèle entraîné
  
  #Cette fonction permet d'évaluer un modèle de classification (clf) en calculant soit :
  #L'accuracy (précision globale) soit  La matrice de confusion ,selon le choix (choice) de l'utilisateur.
  def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test) #calcule la précision moyenne du modèle sur X_test et y_test
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
    

  choix = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)

  clf = prediction(option)
  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))


  #Enregistrement du classifieur clf entrainé (après avoir fait clf.fit()) sous le nom "model" avec Joblib :
  import joblib
  joblib.dump(clf, "model")

  #Enregistrement du classifieur clf entrainé (après avoir fait clf.fit()) sous le nom "model" avec Pickle :
  import pickle
  pickle.dump(clf, open("model", 'wb'))

#Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Joblib :
joblib.load("model")

#Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Pickle :
loaded_model = pickle.load(open("model", 'rb'))
