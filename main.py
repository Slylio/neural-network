import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Génération d'un jeu de données simple
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Création d'un modèle complexe
model = Sequential()
model.add(Dense(200, input_dim=2, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Initialisation de la visualisation
fig, ax = plt.subplots()
plt.ion()

# Préparation de la grille pour la visualisation
xx, yy = np.meshgrid(np.arange(-1.5, 2.5, 0.02),
                     np.arange(-1.0, 1.5, 0.02))

# Initialisation du contour pour l'accès global
contour = ax.contourf(xx, yy, np.zeros_like(xx), alpha=0.8)
scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(y_train, axis=1), edgecolors='k')
scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], c=np.argmax(y_test, axis=1), marker='x')

plt.draw()

# Fonction pour mettre à jour la visualisation
def update_plot(epoch, logs):
    global contour
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Supprimer les anciens contours
    for c in contour.collections:
        c.remove()

    # Redessiner les contours mis à jour
    contour = ax.contourf(xx, yy, Z, alpha=0.8)
    scatter_train.set_offsets(X_train)
    scatter_test.set_offsets(X_test)

    plt.pause(0.01)

# Entraînement du modèle avec visualisation en temps réel
for epoch in range(300):
    model.fit(X_train, y_train, epochs=1, batch_size=10, verbose=0)
    update_plot(epoch, None)

plt.ioff()
plt.show()
