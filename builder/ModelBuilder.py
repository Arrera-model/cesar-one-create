import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    def __init__(self, vectorizer, num_classes,embedding=18):
        self.__vectorizer = vectorizer
        self.__num_classes = num_classes
        self.__embedding_val = embedding
        self.__model = None

    def build(self):
        self.__model = Sequential()

        # 1. Entrée : Texte brut (String)
        self.__model.add(Input(shape=(1,), dtype=tf.string))

        # 2. Prétraitement : On injecte votre vectorizer ici
        self.__model.add(self.__vectorizer)

        # 3. Embedding : Transforme les indices en vecteurs denses
        # input_dim = taille vocabulaire + 1 (pour le token inconnu)
        vocab_size = self.__vectorizer.vocabulary_size()
        self.__model.add(Embedding(input_dim=vocab_size, output_dim=self.__embedding_val))

        # 4. Pooling : Aplati la séquence pour la rendre digeste par les Dense
        self.__model.add(GlobalAveragePooling1D())

        # 5. Couches Denses (Votre architecture originale adaptée)
        self.__model.add(Dense(128, activation='relu'))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(64, activation='relu'))
        self.__model.add(Dropout(0.5))

        # 6. Sortie : Softmax pour la probabilité des classes
        self.__model.add(Dense(self.__num_classes, activation='softmax'))

        # Note : On utilise 'sparse_categorical_crossentropy' car nos labels sont des entiers (0, 1, 2...)
        adam = Adam(learning_rate=0.01)
        self.__model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return self.__model