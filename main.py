import cognitive_package.model.cognitive_classifier_model as cognitive_classifier_model


def main():
    mag_dir = (
        "cognitive_package/res/wordvectors/wiki-news-300d-1m-subword.magnitude"
    )
    models_dir = "cognitive_package/res/pickles"
    print("start...")
    model = cognitive_classifier_model.CognitiveClassifierModel.load_pretrained(
        models_dir, mag_dir
    )
    print("end...")


main()
