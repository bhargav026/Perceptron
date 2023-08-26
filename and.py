from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron
def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND, "y")
    model_and = Perceptron(eta=eta, epochs=epochs)

    model_and.fit(X, y)
    _ = model_and.total_loss()

    model_and.save(filename=modelName, model_dir="model")
    save_plot(df_AND, model_and, filename=plotName)

if __name__ == "__main__":
    AND = {
        "X1":[0,0,1,1],
        "X2":[0,1,0,1],
        "y":[0,0,0,1]
    }
    ETA = 0.1
    EPOCHS = 10
    main(data = AND, modelName = "and.model", plotName= "AND.png", eta =ETA, epochs= EPOCHS)