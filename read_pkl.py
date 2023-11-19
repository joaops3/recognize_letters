import cv2
import numpy as np
import pickle

# Carregando os modelos treinados
with open('mlp_clf.pkl', 'rb') as f:
    mlp_clf = pickle.load(f)

with open('sgd_clf.pkl', 'rb') as f:
    sgd_clf = pickle.load(f)

# Lendo a imagem e convertendo para escala de cinza
image_path = 'dataset/v20220930_partial/i_l/train_69/train_69_00030.png'  # Substitua pelo caminho da sua imagem
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Redimensionando a imagem para um tamanho fixo (ex: 64x64 pixels)
img_resized = cv2.resize(img, (64, 64))

# Achatar a imagem em um vetor unidimensional
img_flattened = img_resized.flatten()

# Fazendo previsões
prediction_mlp = mlp_clf.predict([img_flattened])
prediction_sgd = sgd_clf.predict([img_flattened])

# Imprimindo as previsões
print("MLPClassifier:")
print("A imagem é 'i':", "Sim" if prediction_mlp[0] else "Não")

print("\nSGDClassifier:")
print("A imagem é 'i':", "Sim" if prediction_sgd[0] else "Não")
