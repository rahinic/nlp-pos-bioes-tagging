from torch import nn
import torch

"""RNN Many-to-one multi-class classification neural network model framework design"""

class RNNCompositionNetwork(nn.Module):

    def __init__(self, 
                embedding_dimension, 
                vocabulary_size,
                hidden_dimension,
                num_of_layers,
                dropout,
                output_dimension
                ):
        super(RNNCompositionNetwork, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=embedding_dimension)

        self.lstm = nn.LSTM(embedding_dimension,
                            hidden_dimension,
                            num_of_layers,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dimension*2, output_dimension)

        self.activation_fn = nn.ReLU()


    def forward(self, sample):
        # print(sample.size())
        embedded = self.embedding(sample)
        # print(embedded.size())
        output, (hidden, cell) = self.lstm(embedded)

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # print(hidden.size())

        dense_output = self.fc(hidden)

        #activation function
        outputs=self.activation_fn(dense_output)

        return outputs