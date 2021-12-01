import argparse
import os
import numpy as np
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
import tensorflow as tf

STAGE = "Creating Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    

    ## get the data
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    ## set the seeds
    seed = config["params"]["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    LOSS_FUNCTION = config["params"]["loss_function"]
    METRICS = config["params"]["metrics"]
    OPTIMIZER = config["params"]["optimizer"]
    NUM_CLASSES = config["params"]["num_classes"]

    #Model Creation
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)
    
    #Model training
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        validation_data=VALIDATION_SET,
        verbose=1)

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["base_model_name"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)
    logging.info(f"Evaluation Metrics : {model.evaluate(X_test, y_test)}")
    


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