import tensorflow as tf
import json
import numpy as np
import os
from llama_cpp import Llama
import torch

LISTMODELSUPPROT = [
    "arrera_model_2026",
    "model_gguf"
]

class ArreraIALoad:
    def __init__(self):
        self.__model_type = None
        self.__model = None
        self.__tokeniser = None
        self.__classes = None
        self.__is_loaded = False

    # Methode private

    def __predict_arrera_2026_model(self, sentence: str, confidence_threshold: float = 0.70):
        if not self.__is_loaded:
            return None, 0.0

        input_tensor = tf.constant([sentence], dtype=tf.string)

        if len(input_tensor.shape) == 1:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)

        predictions = self.__model.predict(input_tensor, verbose=0)

        prediction = predictions[0]

        max_index = np.argmax(prediction)
        confidence = prediction[max_index]

        predicted_tag = self.__classes[max_index]

        if confidence < confidence_threshold:
            return None, float(confidence)

        return predicted_tag, float(confidence)

    def __predict_gguf_model(self, prompt, max_tokens=512):
        """
        Génère une réponse. Utilise le format 'chat' compatible OpenAI.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        output = self.__model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )

        return output['choices'][0]['message']['content']

    # Methode public

    def loadArreraModel2026(self, model_path:str, classes_path:str):
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            raise ValueError("Erreur : Fichiers modèles introuvables.")

        try:
            # 1. Chargement du modèle Keras complet (avec la couche TextVectorization incluse)
            self.__model = tf.keras.models.load_model(model_path)

            # 2. Chargement des noms de classes (labels)
            with open(classes_path, 'r', encoding='utf-8') as f:
                self.__classes = json.load(f)

            self.__is_loaded = True
            self.__model_type = LISTMODELSUPPROT[0]
            return True
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du chatbot : {e}")

    def load_model_gguf(self, model_path:str, n_ctx:int=2048):
        try:

            n_gpu_layers = -1 if torch.cuda.is_available() else 0

            self.__model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )

            self.__is_loaded = True
            self.__model_type = LISTMODELSUPPROT[1]
            return True
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement : {e}")

    def send_request(self, sentence: str, confidence_threshold: float = 0.70):
        if self.__model_type == LISTMODELSUPPROT[0]:
            return self.__predict_arrera_2026_model(sentence, confidence_threshold)
        elif self.__model_type == LISTMODELSUPPROT[1]:
            return self.__predict_gguf_model(sentence)
        else:
            return None, 0.0