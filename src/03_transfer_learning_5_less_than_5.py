import argparse
import os
import numpy as np
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, _log_model_summary
import tensorflow as tf

STAGE = "Transfer Learning 5 Less Than 5" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

# Above or equal 5 then 1, else 0
def update_5_less_than_5_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label >= 5
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    

    ## get the data
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    # Binary Outputs (1 for Even, 0 for Odd)
    y_train_bin, y_test_bin, y_valid_bin = update_5_less_than_5_labels([y_train, y_test, y_valid])

    ## set the seeds
    seed = config["params"]["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    base_model_name = config["artifacts"]["base_model_name"]
    #Load the Base Model
    base_model_path = os.path.join(artifacts_dir, model_dir, base_model_name)
    base_model = tf.keras.models.load_model(base_model_path)
    logging.info(f"Loaded Base Model Summary : \n{_log_model_summary(base_model)}")
 
    #Freeze weights
    for layer in base_model.layers[: -1]:
        logging.info(f"trainable status of before {layer.name}:{layer.trainable}")
        layer.trainable = False
        logging.info(f"trainable status of after {layer.name}:{layer.trainable}")

    # Defining new model and compiling it
    NUM_CLASSES_5_less_than_5 = config["params"]["num_classes_transfer_model_5_less_than_5"]
    base_layer = base_model.layers[: -1]
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(NUM_CLASSES_5_less_than_5, activation="softmax", name="output_layer")
    )
    logging.info(f"New model summary: \n{_log_model_summary(new_model)}")
  
    LOSS_FUNCTION = config["params"]["loss_function"]
    METRICS = config["params"]["metrics"]
    OPTIMIZER = config["params"]["optimizer"]
    EPOCHS = config["params"]["epochs_transfer_model_5_less_than_5"]
    VALIDATION_SET = (X_valid, y_valid_bin)

    #Compiling New Model
    new_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    #Model training
    history = new_model.fit(
        X_train, y_train_bin, # << y_train_bin for our usecase
        epochs=EPOCHS, 
        validation_data=VALIDATION_SET, # << y_valid_bin for our usecase
        verbose=1)

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    model_name = config["artifacts"]["transfer_model_name_5_less_than_5"]
    #Save the model
    save_model(new_model, model_name, model_dir_path)
    logging.info(f"Evaluation Metrics : {new_model.evaluate(X_test, y_test_bin)}")
    test_samples = config["params"]["test_sample_in_logs_5_less_than_5"]
    logging.info(f"Actual Outputs : {y_test[:test_samples]}")
    logging.info(f"Actual Binary Outputs : {y_test_bin[:test_samples]}")
    logging.info(f"Predicted Outputs : {np.argmax(new_model.predict(X_test), axis=-1)[:test_samples]}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e