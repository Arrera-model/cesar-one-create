from tensorflow.keras.callbacks import ReduceLROnPlateau
from model_builder_trainer.ModelBuilder import *

class ModelTrainer:

    def __init__(self):
        self.__model = None
        self.__modelLoaded = False
        self.__vectorizer = None

    def create_vectorizer(self,sentences,max_token:int=2000,out_sequence_length:int=20):
        try :
            self.__vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_token,
                output_mode='int',
                output_sequence_length=out_sequence_length
            )
            self.__vectorizer.adapt(sentences)
            return True
        except :
            return False

    def get_vectorizer(self):
        return self.__vectorizer

    def createModel(self,num_classes):
        self.__model = ModelBuilder(vectorizer=self.__vectorizer,
                                   num_classes=num_classes).build()
        self.__modelLoaded = True

    def loadModel(self,model):
        self.__model = model
        self.__modelLoaded = True

    def train(self, train_x, train_y, epochs=150, batch_size=5):
        if self.__modelLoaded:
            lr_schedule = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)

            x_tensor = tf.constant(train_x, dtype=tf.string)

            if len(x_tensor.shape) == 1:
                x_tensor = tf.expand_dims(x_tensor, axis=-1)

            # Entraînement
            history = self.__model.fit(
                x_tensor,
                train_y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[lr_schedule]
            )
            return history
        else :
            return None

    def save(self, filename='arrera_model.keras'):
        try :
            self.__model.save(filename)
            return True
        except :
            return False