from torch.utils.data import DataLoader
from model import RNNCompositionNetwork
from dataset import composeDataset, compositionNetworksTagSets
import gzip, pickle, torch

############################ 01. Fetch Validation Dataset ################################
validation_dataset = DataLoader(dataset=composeDataset(myDataset="validation"),
                                batch_size=16,
                                shuffle=False)

################################ 02. Look-up tables ######################################
# (a) targets:-
dict1 = compositionNetworksTagSets
composition_tags, composition_tags_reverse = dict1.dictionary()
# (b) samples and pos tags:-
def fetchWordsAndPOSLookupTables():
    def load_pickles(filename):
        data_raw_filepath = "data/raw/"

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
        # (i) Vocabulory: 
    nn1_pos_reverse = load_pickles(list_of_lookup_tables[0])
    nn1_pos = load_pickles(list_of_lookup_tables[1])
    nn1_vocab = load_pickles(list_of_lookup_tables[2])
    nn1_vocab_reverse = load_pickles(list_of_lookup_tables[3])

    return nn1_vocab, nn1_pos, nn1_vocab_reverse, nn1_pos_reverse


penn_vocab, penn_pos_tags_reverse, penn_vocab_reverse, penn_pos_tags = fetchWordsAndPOSLookupTables()

############################### 03. Hyperparameters #########################################

VOCAB_SIZE = len(penn_vocab)+len(penn_pos_tags)+1
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_OF_CLASSES = len(composition_tags)
EPOCHS = 10
LEARNING_RATE = 0.02
BATCH_SIZE = 64

################################### 04. Load NN model  ########################################
print("03. builing the model...")
model = RNNCompositionNetwork(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0.2,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")

################################# 05. Load model weights ######################################
model.load_state_dict(torch.load("notebooks/composition_networks/saved_weights.pt"))
model.eval()

################################# 06. Sample predictions ######################################

for idx, sample in enumerate(validation_dataset):
#   if idx>3:
#     break

  examples = sample[1][:5]
  actual_labels = sample[0][:5]
print('-'*100)
print("Let us see few examples of our trained models' predictions...")
# print(examples)
# print(actual_labels)

with torch.no_grad():
  output = model(examples)
  
  rounded_preds = torch.round(output)
  
  _,pred_label = torch.max(rounded_preds, dim = 1)
  
  print(f"Actual labels: {actual_labels}")
  print(f"Predicted labels: {pred_label}")
  print('-'*100)

############################### 07. Calculate validation dataset accuracy ########################

overall_accuracy = []
total_samples = len(validation_dataset)*16
print(total_samples)
for idx, sample in enumerate(validation_dataset):
  
  actual_label = sample[0]
  with torch.no_grad():
    predicted_label = model(sample[1])
    rounded_preds = torch.round(predicted_label)
    _, pred_label = torch.max(rounded_preds, dim=1)

    correct = (pred_label == actual_label).float()

    overall_accuracy.append(torch.sum(correct))


overall_accuracy_final = []
for item in overall_accuracy:
  overall_accuracy_final.append(int(item))

print(f"Total validation dataset accuracy of our trained model is: {(sum(overall_accuracy_final)/total_samples)*100}%")
print('='*100)

################################ 08. Convert idx to text and export ##################################