from torch import nn
import torch

"""RNN Many-to-many multi-class classification neural network model structure definition"""

class RNNBIOESTagger(nn.Module):

    def __init__(self, 
                embedding_dimension, 
                vocabulary_size,
                hidden_dimension,
                num_of_layers,
                dropout,
                output_dimension
                ):
        super(RNNBIOESTagger, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=embedding_dimension)

        self.lstm = nn.LSTM(embedding_dimension,
                            hidden_dimension,
                            num_of_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_dimension*2, output_dimension)

        self.activation_fn = nn.Tanh()


    def forward(self, sample):

        # (1)- Embedding layer
        embedded = self.embedding(sample)

        #(2)- LSTM layer 1
        output, (hidden, cell) = self.lstm(embedded)       

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-1,:,:], hidden[0,:,:]), dim = 1)

        #(3)- LSTM to linear layer: Final set of tags
        dense_output = self.fc(output)        

        #activation function
        outputs=self.activation_fn(dense_output)
 
        return outputs