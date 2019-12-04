from keras.models import model_from_json
from Interpreter import Interpreter
from Graphs import Graphs
import numpy as np
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    """
    Evaluates model based on test dataset,
    from .json and  .h5 models.
    """
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 2
    IMAGE_SHAPE = (128, 128, 1)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE,
        EPOCHS,
        TARGET_SIZE
    )

    train_images, validation_images, test_images = inter.split_data()

    # Load .json file and create model.
    json_file = open('model_simple.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights.
    loaded_model.load_weights('model_simple.h5')

    pred = loaded_model.predict(
        test_images
    )
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    print(accuracy_score(
        test_images.classes,
        pred
    ))

    graphs = Graphs()
    graphs.show_confusion_matrix(
        test_images.classes,
        pred,
        np.array(['glaucoma', 'non-glaucoma'])
    )

    # TODO: Represents ROC curve could be better...
    # Ref.: https://github.com/GustavoMourao/heart_dis_classify/blob/master/metric_eval_response_ROC.py
