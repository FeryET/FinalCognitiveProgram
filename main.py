import cognitive_package.model.cognitive_classifier_model as cognitive_classifier_model
import cognitive_package.controller.cognitive_controller as cognivite_controller
import sys


def main():
    mag_dir = (
        "cognitive_package/res/wordvectors/wiki-news-300d-1m-subword.magnitude"
    )
    models_dir = "cognitive_package/res/pickles"
    texts_path = "../../datasets_of_cognitive/Data/Unprocessed Data/"
    print("start...")
    controller = cognivite_controller.CognitiveController(
        texts_path, mag_dir, models_dir, sys.argv
    )
    sys.exit(controller.exec_())
    print("end...")


main()
