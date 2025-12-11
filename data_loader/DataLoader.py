import json
import numpy as np
import os

class DataLoader:
    def __init__(self, json_file:str):
        self.file_path = json_file
        self.__classes = None
        self.__labels = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Erreur : Le fichier '{self.file_path}' est introuvable.")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"Erreur : Le fichier '{self.file_path}' n'est pas un JSON valide.")

        sentences = []
        self.__labels = []
        self.__classes = []

        for intent in data['intents']:
            tag = intent['tag']

            if tag not in self.__classes:
                self.__classes.append(tag)


            for pattern in intent['patterns']:
                sentences.append(pattern)
                self.__labels.append(tag)

        self.__classes = sorted(self.__classes)

        return sentences, np.array(self.__labels), self.__classes

    def save_classe_file(self,class_file:str="classes.json"):
        try :
            with open(class_file, 'w', encoding='utf-8') as file:
                json.dump(self.__classes, file)
            return True
        except :
            return False

    def encoding_label(self):
        label_array = np.array(self.__labels)
        label_to_index = {label: i for i, label in enumerate(self.__classes)}
        train_y = np.array([label_to_index[l] for l in label_array])
        return train_y