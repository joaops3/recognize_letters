from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
import os
import cv2
import numpy as np
import pickle

# Definindo o caminho do dataset
dataset_path = 'dataset/v20220930_partial'

# Mapeando as classes para números
class_mapping = {'A_u': "a", 'E_u': "e", 'I_u': "i", 'O_u': "o", 'U_u': "u",
                 'a_l': "a", 'e_l': "e", 'i_l': "i", 'o_l': "o", 'u_l': "u"}

# Preparando os dados
data = []
labels = []

for class_folder in os.listdir(dataset_path):
    class_folder_path = os.path.join(dataset_path, class_folder)
    for image_folder in os.listdir(class_folder_path):
        image_folder_path = os.path.join(class_folder_path, image_folder)
        for image_file in os.listdir(image_folder_path):
            image_file_path = os.path.join(image_folder_path, image_file)
            # Lendo a imagem e convertendo para escala de cinza
            img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
            # Redimensionando a imagem para um tamanho fixo (ex: 64x64 pixels)
            img_resized = cv2.resize(img, (64, 64))
            # Achatar a imagem em um vetor unidimensional
            img_flattened = img_resized.flatten()
            data.append(img_flattened)
            labels.append(class_mapping[class_folder])

# Convertendo listas para arrays numpy
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

y_train_I = (y_train=="i")
y_test_I = (y_test=="i")

# Treinando o MLPClassifier
mlp_clf = MLPClassifier(random_state=42)
mlp_clf.fit(X_train, y_train_I)

# Salvando o modelo MLPClassifier
with open('mlp_clf.pkl', 'wb') as f:
    pickle.dump(mlp_clf, f)

# Treinando o SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_I)

# Salvando o modelo SGDClassifier
with open('sgd_clf.pkl', 'wb') as f:
    pickle.dump(sgd_clf, f)

# Fazendo previsões
y_train_pred_mlp = cross_val_predict(mlp_clf, X_train, y_train_I, cv=3)
y_train_pred_sgd = cross_val_predict(sgd_clf, X_train, y_train_I, cv=3)

# Calculando as pontuações
accuracy_mlp = accuracy_score(y_train_I, y_train_pred_mlp)
precision_mlp = precision_score(y_train_I, y_train_pred_mlp, zero_division=1)
recall_mlp = recall_score(y_train_I, y_train_pred_mlp)
f1_mlp = f1_score(y_train_I, y_train_pred_mlp)

accuracy_sgd = accuracy_score(y_train_I, y_train_pred_sgd)
precision_sgd = precision_score(y_train_I, y_train_pred_sgd, zero_division=1)
recall_sgd = recall_score(y_train_I, y_train_pred_sgd)
f1_sgd = f1_score(y_train_I, y_train_pred_sgd)

print("MLPClassifier:")
print(f"Acurácia: {accuracy_mlp*100:.2f}%")
print(f"Precisão: {precision_mlp*100:.2f}%")
print(f"Recall: {recall_mlp*100:.2f}%")
print(f"F1 Score: {f1_mlp*100:.2f}%")

print("\nSGDClassifier:")
print(f"Acurácia: {accuracy_sgd*100:.2f}%")
print(f"Precisão: {precision_sgd*100:.2f}%")
print(f"Recall: {recall_sgd*100:.2f}%")
print(f"F1 Score: {f1_sgd*100:.2f}%")
