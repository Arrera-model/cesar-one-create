from builder.ModelBuilder import ModelBuilder
from data_loader.DataLoader import DataLoader
from testeur.ArreraIALoad import ArreraIALoad
from trainer.ModelTrainer import ModelTrainer

def main():
    print("Programme pour crée et entrainer le model cesar one")
    v = int
    while v !=0:
        try :
            v = int(input("1. Create le model\n"
                          "2.Entrainer le model\n"
                          "3.Essayer le model\n"
                          "0.Quitter\n$ "))
            match v :
                case 1 :
                    try :
                        print("create le model")
                        path = input("Emplacement des donnée de pre-entrainement :  ")
                        classe_file_path = input("Emplacement des classes :  ")
                        modelTrainer = ModelTrainer()
                        dataLoader = DataLoader(path)
                        sentences, labels, classes = dataLoader.load_data()
                        if not dataLoader.save_classe_file(classe_file_path):
                            return
                        modelTrainer.create_vectorizer(sentences)
                        modelTrainer.createModel(len(classes))
                        modelTrainer.train(sentences,dataLoader.encoding_label())
                        modelTrainer.save("model/arrera-cesar-one.keras")
                        print("Model crée et pre-entrainer")
                    except:
                        print("Erreur")
                case 2 :
                    try :
                        print("Entrainer le model")
                        path = input("Emplacement des donnée de entrainement :  ")
                        modelPath = input("Emplacement du model :  ")
                        modelTrainer = ModelTrainer()
                        dataLoader = DataLoader(path)
                        sentences, labels, classes = dataLoader.load_data()
                        modelTrainer.loadModel(modelPath)
                        modelTrainer.train(sentences,dataLoader.encoding_label())
                        modelTrainer.save("model/arrera-cesar-one.keras")
                        print("Model entrainé")
                    except:
                        print("Erreur")
                case 3 :
                    try :
                        print("Essayer le model")
                        modelPath = input("Emplacement du model : ")
                        classe_file_path = input("Emplacement des classes :  ")
                        arreraIALoad = ArreraIALoad()
                        arreraIALoad.loadArreraModel2026(modelPath,classe_file_path)
                        print("\n--- Chatbot Prêt ! (Tapez 'quit' pour quitter) ---")
                        while True:
                            user_input = input("Vous : ")
                            if user_input.lower() == 'quit':
                                break
                            tag, confidence = arreraIALoad.send_request(user_input, confidence_threshold=0.70)

                            if tag:
                                print(f"Bot: [Intention: {tag}] ({confidence:.2%})")
                                # Ici, plus tard, vous ajouterez la logique de réponse :
                                # if tag == "salutation": print("Bonjour !")
                            else:
                                print(f"Bot: Je n'ai pas compris... ({confidence:.2%})")
                    except :
                        print("Erreur")
                case 0 :
                    print("bye")
        except:
            print("Valeur invalide")

if __name__ == "__main__":
    main()