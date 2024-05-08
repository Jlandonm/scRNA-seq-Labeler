import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import umap
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the data
def process_quarter(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Remove header row and column
    data = data.iloc[1:]
    data = data.iloc[:, 1:]
    data = data.T

    # Transpose data so that cells are rows and genes are columns

    return data

# quarter_files = ["/content/drive/MyDrive/datasets/quarter_1.csv", "/content/drive/MyDrive/datasets/quarter_2.csv", "/content/drive/MyDrive/datasets/quarter_3.csv", "/content/drive/MyDrive/datasets/quarter_4.csv", "/content/drive/MyDrive/datasets/quarter_5.csv", "/content/drive/MyDrive/datasets/quarter_6.csv", "/content/drive/MyDrive/datasets/quarter_7.csv", "/content/drive/MyDrive/datasets/quarter_8.csv"]
quarter_files = ["/content/drive/MyDrive/datasets/quarter_1.csv", "/content/drive/MyDrive/datasets/quarter_2.csv"]

latent_dim = 10


class VAE(Model):
    def __init__(self, latent_dim, input_shape):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(input_shape, activation='sigmoid'),
        ])

    def call(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(log_var * 0.5) + mean
        reconstructed = self.decoder(z)
        return reconstructed

# input_shape = X_train_pca.shape[1]
vae = VAE(latent_dim, 10)
optimizer = tf.keras.optimizers.Adam(learning_rate=.003)
vae.compile(optimizer=optimizer, loss='categorical_focal_crossentropy')

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True)
X_test_concatenated = []
y_test_concatenated = []


for quarter_file in quarter_files:
    quarter_data = process_quarter(quarter_file)
    X = quarter_data.iloc[:, :-1].values  # Features (gene expression data)
    y = quarter_data.iloc[:, -1].values   # Labels (cell types)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #PCA
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)

    ump = umap.UMAP(n_components=10)
    X_train_pca = ump.fit_transform(X_train_pca)

    # Fit the VAE model
    history = vae.fit(X_train_pca, y_train, epochs=25, batch_size=32, validation_split=0.2)
    X_test_concatenated.append(X_test_scaled)
    y_test_concatenated.append(y_test)


X_test_concatenated = np.concatenate(X_test_concatenated)
y_test_concatenated = np.concatenate(y_test_concatenated)


X_train, X_test, y_train, y_test = train_test_split(X_test_concatenated, y_test_concatenated, test_size=0.25, random_state=42)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

ump = umap.UMAP(n_components=10)
X_train_pca = ump.fit_transform(X_train_pca)
X_test_pca = ump.transform(X_test_pca)


# Use VAE to generate latent representations
X_train_latent = vae.encoder(X_train_pca).numpy()
X_test_latent = vae.encoder(X_test_pca).numpy()

# Train classifier (Random Forest)
classifier = RandomForestClassifier(n_estimators=250, random_state=42)
classifier.fit(X_train_latent, y_train)

# Predict
y_pred = classifier.predict(X_test_latent)

print(classification_report(y_test, y_pred))



vae.save("vae_model2.keras")

from joblib import dump
dump(classifier, 'random_forest_model.joblib')
