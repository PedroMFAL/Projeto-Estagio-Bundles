# Importa as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Carrega os dados de um arquivo pickle
ds = pd.read_pickle("data.pkl")

# Escala os recursos (features)
scaler = MinMaxScaler()
name_column = ds["name"]  # Salva a coluna 'name'
ds = ds.drop("name", axis=1)  # Remove a coluna 'name' temporariamente
y = ds["best_algorit"]  # Separa a coluna de alvo (target)
ds = ds.drop("best_algorit", axis=1)  # Remove a coluna de alvo
ds = pd.DataFrame(scaler.fit_transform(ds), columns=ds.columns)  # Aplica a normalização MinMax
ds["best_algorit"] = y  # Re-adiciona a coluna de alvo
ds["name"] = name_column  # Re-adiciona a coluna 'name'
label_encoder = LabelEncoder()
ds['best_algorit'] = label_encoder.fit_transform(ds['best_algorit'])  # Codifica os labels

# funcao para separar dados de teset e treino
def seperate_original(data, number):
    lis = ["MealRec", "NetEase", "iFashion", "Steam", "Youshu"]
    nao = data[data['name'].str.contains(lis[number], case=False, na=False)].index
    train = data.drop(nao)
    test = data[data['name'] == lis[number]]
    return train, test

# Função para apenas usar os datasets originais para teste e treino
def original(data, number):
    lis = ["MealRec", "NetEase", "iFashion", "Steam", "Youshu"]
    test = data[data['name'] == lis[number]]
    lis.pop(number)
    train = data[data['name'].isin(lis)]
    return train, test



#calcular utilizando os datasetoids
# Avaliação com Linear Discriminant Analysis (LDA)
soma = 0
for i in range(5):
    clf = LinearDiscriminantAnalysis()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Linear Discriminant Analysis (LDA)")
print(soma / 5 * 100)

# Avaliação com Naive Bayes
soma = 0
for i in range(5):
    clf = GaussianNB()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Naive Bayes")
print(soma / 5 * 100)

# Avaliação com Gradient Boosting Machine (GBM)
soma = 0
for i in range(5):
    clf = GradientBoostingClassifier()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Gradient Boosting Machine (GBM)")
print(soma / 5 * 100)

# Avaliação com Regressão Logística
soma = 0
for i in range(5):
    clf = LogisticRegression()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("LogisticRegression")
print(soma / 5 * 100)

# Avaliação com Random Forest
soma = 0
for i in range(5):
    clf = RandomForestClassifier()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("random forest")
print(soma / 5 * 100)

# Avaliação com SVC
soma = 0
for i in range(5):
    clf = svm.SVC()
    train, test = seperate_original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("SVC")
print(soma / 5 * 100)

print("------------------------------------")

# fazer sem os datasetoids
for i in range(5):
    clf = LinearDiscriminantAnalysis()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Linear Discriminant Analysis (LDA)")
print(soma / 5 * 100)

# Avaliação com Naive Bayes
soma = 0
for i in range(5):
    clf = GaussianNB()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Naive Bayes")
print(soma /5 * 100)

# Avaliação com Gradient Boosting Machine (GBM)
soma = 0
for i in range(5):
    clf = GradientBoostingClassifier()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy()
    y_test = test["best_algorit"]
    
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test.reshape(1, -1))
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("Gradient Boosting Machine (GBM)")
print(soma / 5 * 100)

# Avaliação com Regressão Logística
soma = 0
for i in range(5):
    clf = LogisticRegression()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("LogisticRegression")
print(soma / 5 * 100)

# Avaliação com Random Forest
soma = 0
for i in range(5):
    clf = RandomForestClassifier()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("random forest")
print(soma / 5 * 100)

# Avaliação com SVC
soma = 0
for i in range(5):
    clf = svm.SVC()
    train, test = original(ds, i)
    train = pd.DataFrame(train)
    train = train.drop("name", axis=1)
    x_train = train.drop("best_algorit", axis=1).to_numpy()
    y_train = train["best_algorit"].to_numpy()
    x_test = test.drop(["best_algorit", "name"], axis=1).to_numpy().reshape(1, -1)
    y_test = test["best_algorit"]
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    
    # Calcula a acurácia
    accuracy = accuracy_score(y_test, predictions)
    soma += accuracy

print("SVC")
print(soma / 5 * 100)