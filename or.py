from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron

OR = {
    "X1":[0,0,1,1],
    "X2":[0,1,0,1],
    "y":[0,1,1,1]
}
df_OR = pd.DataFrame(OR)

X, y = prepare_data(df_OR,"y" )

ETA = 0.1
EPOCHS = 10

model_or = Perceptron(eta =ETA, epochs= EPOCHS)

model_or.fit(X,y)
_ = model_or.total_loss()

model_or.save(filename = "or.model",model_dir = "model")

save_plot(df_OR, model_or, filename='OR.png')