from torch import nn
import torch

"""RNN Sentence chunks classification (NP/VP/PP etc.) NN model"""

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
        
        embedded = self.embedding(sample) #embedding layer
        
        output, (hidden, cell) = self.lstm(embedded)

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-1,:,:], hidden[0,:,:]), dim = 1)
        

        dense_output = self.fc(hidden)

        #activation function
        outputs=self.activation_fn(dense_output)

        return outputs