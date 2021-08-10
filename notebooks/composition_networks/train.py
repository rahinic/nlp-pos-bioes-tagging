from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch
from dataset import composeDataset
from dictionary import compositionNetworksTagSets
from model import RNNCompositionNetwork
import pickle, time, gzip
##########################################################################################
################################# 01.TRain/Test Dataset ##################################

train_dataset = DataLoader(dataset=composeDataset(myDataset="train"),
                                batch_size=16,
                                shuffle=True)
                                
test_dataset = DataLoader(dataset=composeDataset(myDataset="test"),
                                batch_size=16,
                                shuffle=True)
##########################################################################################
################################ 02. Look-up tables ######################################
# (a) targets:-
dict1 = compositionNetworksTagSets
composition_tags, composition_tags_reverse = dict1.dictionary()
# (b) samples and pos tags:-
def fetchWordsAndPOSLookupTables():
    def load_pickles(filename):
        data_raw_filepath = "C:/Users/rahin/projects/nlp-pos-bioes-tagging/data/raw/"

        if "pklz" in filename:     
            file = gzip.open(data_raw_filepath+str(filename),"rb")
        else:
               # print(f"loading: {data_raw_filepath+str(filename)}")
            file = open(data_raw_filepath+str(filename),"rb")

        return pickle.load(file)

    list_of_lookup_tables = ["nn1_wsj_idx_to_pos.pkl",
                         "nn1_wsj_pos_to_idx.pkl",
                         "nn1_wsj_word_to_idx.pkl",
                         "nn1_wsj_idx_to_word.pkl"]
        # (a) Vocabulory: 
    nn1_pos_reverse = load_pickles(list_of_lookup_tables[0])
    nn1_pos = load_pickles(list_of_lookup_tables[1])
    nn1_vocab = load_pickles(list_of_lookup_tables[2])
    nn1_vocab_reverse = load_pickles(list_of_lookup_tables[3])

    return nn1_vocab, nn1_pos, nn1_vocab_reverse, nn1_pos_reverse


penn_vocab, _, _, penn_pos_tags = fetchWordsAndPOSLookupTables()

##########################################################################################
################################# 02.Model Parameters ####################################
VOCAB_SIZE = len(penn_vocab)+len(penn_pos_tags)+1
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_OF_CLASSES = len(composition_tags)
EPOCHS = 10
LEARNING_RATE = 0.02
BATCH_SIZE = 16
################################### 02. NN Model  #######################################
print("03. builing the model...")
model = RNNCompositionNetwork(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")
##########################################################################################
############################# 03. Optimizer and Loss  #################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss() #ignore_index=46393

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    # correct = (rounded_preds == y).float() 
    _,pred_label = torch.max(rounded_preds, dim = 1)

    # sanity check:
    # print(f"Actual label: {y}")
    # print(rounded_preds)
    # print(f"Predicted label: {pred_label}")
    correct = (pred_label == y).float()
    acc = correct.sum() / len(correct)
    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

##########################################################################################
############################## 04. NN Model Train Definition #############################

def train(model, dataset, optimizer, criterion):
    
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for idx, sample in enumerate(dataset):
       
       current_samples = sample[1]
       current_labels = sample[0]
    
       optimizer.zero_grad()

       predicted_labels = model(current_samples)

       
       loss = criterion(predicted_labels, current_labels)
       accuracy = binary_accuracy(predicted_labels, current_labels)

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

##########################################################################################
################################ 05. NN Model Eval Definition ############################
def evaluate(model, dataset, criterion):
    
    t = time.localtime()
    start_time = time.strftime("%H:%M:%S", t)
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for idx, sample in enumerate(dataset):
            current_samples = sample[1]
            current_labels = sample[0]

            predicted_labels = model(current_samples)

            loss = criterion(predicted_labels, current_labels)
            accuracy = binary_accuracy(predicted_labels, current_labels)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

############################################################################################
################################## 06. NN Model training #####################################
N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    print(epoch)
     
    #train the model
    train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, test_dataset, criterion)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print("-------------------------------------------------------------------")
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print("-------------------------------------------------------------------")
    