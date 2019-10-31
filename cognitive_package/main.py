import model.cognitive_classifier_model

# TODO: remove pymagnitude from the project.
def main():
    mag_dir = "Resources/WordVectors/wiki-news-300d-1m-subword.magnitude"
    models_dir = "Resources/Pickles/"
    print("start...")
    model = model.cognitive_classifier_model.CognitiveClassifierModel.load_pretrained(
        models_dir, mag_dir
    )
    print("end...")


main()
