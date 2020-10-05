#import os
#import time
import uuid
#import joblib

#from nntraining.config import INPUT_STRING, OUTPUT_STRING
#from nntraining.main.dataStorage import DataStorage
from nntraining.main.nn import import_scaler_and_model#, create_sequential_model, print_model_information, create_callbacks, \
   # plot_training_history


class NeuralNetwork(object):
    """Class for neural networks. It contains the actual model, the scaler, the possibility to create a sequential
     model and train it. It is possible to predict a single data point or multiple data frames given in a list"""

    def __init__(self, input_scaler=None, output_scaler=None, model=None, model_name=uuid.uuid4().hex):
        self.input_scaler, self.output_scaler, self.model = input_scaler, output_scaler, model
        self.model_name = model_name

    def predict_data_point(self, input_vector):
        """Predict a single data point. The scaler within this class is used to scale the inputs"""
        # use standard scaler to scale data and predict fatigue life
        input_vector_scaled = self.input_scaler.transform(input_vector)
        prediction_scaled = self.model.predict(input_vector_scaled)
        prediction = self.output_scaler.inverse_transform(prediction_scaled)
        return prediction

    def load_existing_model(self, path, model_name):
        self.model, self.input_scaler, self.output_scaler = import_scaler_and_model(path, model_name)
    """
    def save_model_and_scaler(self, path, model_name):
        try:
            self.model.save(os.path.join(path, model_name + ".h5"))
            joblib.dump(self.input_scaler, os.path.join(path, model_name + "_input.save"))
            joblib.dump(self.output_scaler, os.path.join(path, model_name + "_output.save"))
        except (FileExistsError, OSError) as e:
            print("Error while saving model and scaler:", e)

    def create_sequential_model(self, neurons, number_of_layers, l2_regularization,
                                learning_rate, beta_1, activation_hidden_layer, n_input, n_output):

        self.model = create_sequential_model(neurons, number_of_layers, l2_regularization,
                                             learning_rate, beta_1, activation_hidden_layer, n_input, n_output)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size, verbose=2,
                    early_stopping=False, csv_logger=False, checkpoint=False, tensorboard=False, tmp_folder=None):

        print_model_information(self.model, verbose)

        # Train the model, use batches of n samples, verbose = 1: progress bar
        hist = self.model.fit(x_train, y_train, validation_data=[x_val, y_val],
                              batch_size=batch_size, epochs=epochs, verbose=verbose,
                              callbacks=create_callbacks(tmp_folder=tmp_folder, name=self.model_name,
                                                         early_stopping=early_stopping, csv_logger=csv_logger,
                                                         checkpoint=checkpoint, tensorboard=tensorboard))
        plot_training_history(hist, verbose)

        # calculate losses
        print("Losses are:", self.model.metrics_names)
        train_loss = self.model.evaluate(x=x_train, y=y_train, verbose=0)
        val_loss = self.model.evaluate(x=x_val, y=y_val, verbose=0)
        print("Train Losses:", train_loss)
        print("Validation Losses:", val_loss)
        return hist, train_loss, val_loss

    # def predict_df(self, list_of_df):"""
    #     """Predict multiple dataframes"""
    #
    #     if not isinstance(list_of_df, list):
    #         raise TypeError("Argument must be a list of dataframes.")
    #
    #     for df in list_of_df:
    #         # extend data frame
    #         for output in OUTPUT_STRING:
    #             df["NN_" + output] = None
    #         df.reset_index(drop=True, inplace=True)
    #
    #         batchsize = 1000000
    #         max_index = df.__len__() / batchsize
    #         # generate data batch by batch
    #         for idx in range(int(ceil(max_index))):
    #             if idx + 1 > max_index:
    #                 upper_limit = df.__len__() + 1
    #             else:
    #                 upper_limit = (idx + 1) * batchsize
    #
    #             # use NN to calculate Data
    #             nn = self.predict_fatigue_life(df.loc[idx * batchsize:upper_limit, INPUT_STRING].values)
    #             df.loc[idx * batchsize:upper_limit, "NN_" + OUTPUT_STRING[0]] = nn
    #
    #         for output in OUTPUT_STRING:
    #             if output in df.columns:
    #                 df["Delta_" + output] = abs(df["NN_" + output] - df[output])
    #                 df['Error in percent_' + output] = (abs(df["Delta_" + output]) / df[output]) * 100
    #                 df["Delta_" + output] = pd.to_numeric(df["Delta_" + output])
    #                 df['Error in percent_' + output] = pd.to_numeric(df['Error in percent_' + output])

"""
if __name__ == "__main__":
    path = "./data.h5"  #"E:\Kai\LandingRockets/data.h5"
    storage = DataStorage(path_to_data=path)
    storage.pre_process_data(input_string=INPUT_STRING, output_string=OUTPUT_STRING)
    model = NeuralNetwork(input_scaler=storage.input_scaler, output_scaler=storage.output_scaler,
                          model_name=time.strftime("%Y%m%d-%H%M%S") + "_len_" + str(storage.size))
    model.create_sequential_model(neurons=1024, number_of_layers=3, l2_regularization=0.000001,
                                  learning_rate=0.001, beta_1=0.1, activation_hidden_layer="LeakyReLU",
                                  n_input=len(INPUT_STRING), n_output=len(OUTPUT_STRING))
    hist, val_loss, train_loss = model.train_model(x_train=storage.train["x"], y_train=storage.train["y"],
                                                   x_val=storage.val["x"], y_val=storage.val["y"],
                                                   epochs=500, batch_size=1024, verbose=2)

    model.save_model_and_scaler(path="D:/07_Forschung/LandingRockets", model_name="140k_samples_1024neurons_3layers_l2-0.000001")
"""