from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador SVM
clf = svm.SVC(kernel='linear')

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predecir las etiquetas de los datos de prueba
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
accuracy = clf.score(X_test, y_test)
print("Precisión del modelo SVM:", accuracy)
