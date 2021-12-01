import tensorflow as tf
import os
import time
import logging
import io      # <<<<To log our model summary

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

## log our model summary information in logs
def _log_model_summary(model):
    with io.StringIO() as stream:
        model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
        summary_str = stream.getvalue()
    return summary_str

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, name="hiddenLayer1"),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(100,  name="hiddenLayer2"),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    # model_clf.summary()
    return model_clf ## <<< untrained model


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)
    logging.info(f"Base Model is Saved at {path_to_model}")