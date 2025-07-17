import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import optimizer
from torch.utils.data import TensorDataset, DataLoader


#subclassing torch module -> class inheritance basically

class LSTM(nn.Module):
    #input_size -> number of features at each time step
    #hidden_dim -> size of hidden state vector
    #output_size -> number of feature predictions (aka. how many features am I predicting (1 --> sales in this case))
    #n_layers -> number of inner layers in the black box
    def __init__(self,input_size,hidden_size,output_size, n_layers ):
        #call methods from parent function
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        #batch_first=True -> changes whether batch size or sequence length is first
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)

    #forward pass for the model, predicts the final embedding
    def forward(self,x):
        #FIRST WAY
        #lstm_out, _ = self.lstm(x)
        #final hidden state output
        #output = self.fc(lstm_out[:, -1, :])
        # return output

        #SECOND WAY
        #Hidden state 0
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).requires_grad_()
        #Cell state 0
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).requires_grad_()

        lstm_out, (h, c) = self.lstm(x, (h0.detach(), c0.detach()))

        output = self.fc(lstm_out[:, -1, :])

        return torch.sigmoid(output)

    def main(self):
            #there are never any promotions
                # d type data types, parse_dates -> convert column to datetime objects
            #this creates a data frame
        train = pd.read_csv("train.csv", usecols=['date','store_nbr', 'family', 'sales'],
                         dtype={'store_nbr': 'category', 'family': 'category', 'sales': 'float64'},
                        parse_dates=['date']
                        )
        test = pd.read_csv( "test.csv", usecols=['date','store_nbr', 'family'],
                            dtype={'store_nbr': 'category', 'family': 'category'},
                            parse_dates=['date']
                             )

        #print first 5 rows
        print(train.head())
        print(test.head())

        #concatinate together to have preprocessing on each of these
        df = pd.concat([train, test])
        df['date'] = pd.to_datetime(df['date'])

        #convert to datetime64 object
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
            #weekday and weekend might not be necessary
        df['weekday'] = df['date'].dt.dayofweek
        df['weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df = df.drop('date', axis=1)

        print(df.head())

        #split back
        train = df.loc[train.index, :]
        test = df.loc[test.index, :]
        test.drop('sales', axis=1, inplace=True)

        #clean training dataset where sales aren't there
        clean = train.dropna(subset=['sales'])

        features = clean.columns.difference(['date'])
        x = clean[features].values
        y = clean['sales']

        model =  LSTM(4,100,1,1)
            #mean squared error
        criterion = nn.MSELoss()
        learning_rate = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        #Prepare data
        train_dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dl = DataLoader(train_dataset, batch_size=64, shuffle=True)

        #write training loop
        for epoch in range(10):
            running_loss = 0.0

            for i, data in enumerate(,0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                #loss
                print()



    if __name__ == "__main__":
        main()
