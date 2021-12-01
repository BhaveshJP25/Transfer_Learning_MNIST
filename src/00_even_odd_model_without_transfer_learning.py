import argparse
import os
import numpy as np
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, _log_model_summary
import tensorflow as tf

STAGE = "Even Odd Model Without Transfer Learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

# 1 for Even, 0 for Odd
def update_even_odd_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        even_condition = label%2 == 0
        list_of_labels[idx] = np.where(even_condition, 1, 0)
    return list_of_labels

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    

    ## get the data
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    # Binary Outputs (1 for Even, 0 for Odd)
    y_train_bin, y_test_bin, y_valid_bin = update_even_odd_labels([y_train, y_test, y_valid])

    ## set the seeds
    seed = config["params"]["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    LOSS_FUNCTION = config["params"]["loss_function"]
    METRICS = config["params"]["metrics"]
    OPTIMIZER = config["params"]["optimizer"]
    NUM_CLASSES = config["params"]["num_classes_scratch_bin"]

    #Model Creation
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    # Logging Model Summary
    logging.info(f"Scratch Bin Model Summary : \n{_log_model_summary(model)}")

    EPOCHS = config["params"]["epochs_transfer_scratch_bin"]
    VALIDATION_SET = (X_valid, y_valid_bin)
    
    #Model training
    history = model.fit(
        X_train, y_train_bin, 
        epochs=EPOCHS, 
        validation_data=VALIDATION_SET,
        verbose=1)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["scratch_bin_model_name"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)
    logging.info(f"Evaluation Metrics : {model.evaluate(X_test, y_test_bin)}")
    test_samples = config["params"]["test_sample_in_logs_scratch_bin"]
    logging.info(f"Actual Outputs : {y_test[:test_samples]}")
    logging.info(f"Actual Binary Outputs : {y_test_bin[:test_samples]}")
    logging.info(f"Predicted Outputs : {np.argmax(model.predict(X_test), axis=-1)[:test_samples]}")
    


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