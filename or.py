from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron
import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok =True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format = '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a')
def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"this is the Raw dataset: \n{df}")
    X, y = prepare_data(df, "y")
    model = Perceptron(eta=eta, epochs=epochs)

    model.fit(X, y)
    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df, model, filename=plotName)

if __name__ == "__main__":
    OR = {
        "X1":[0,0,1,1],
        "X2":[0,1,0,1],
        "y":[0,1,1,1]
    }
    ETA = 0.1
    EPOCHS = 10
    try:
        logging.info(f"**********Starting Training for OR **************")
        main(data = OR, modelName = "or.model", plotName= "OR.png", eta =ETA, epochs= EPOCHS)
        logging.info(f"*******Done Training for OR ***********")

    except Exception as e:
        logging.exception(e)
        raise e