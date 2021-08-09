
from dictionary import compositionNetworksTagSets, compositionNetworksDataset
import pickle, gzip
from typing import Tuple, List
import torch
from torch.utils.data import Dataset,DataLoader

class composeDataset(Dataset):

    # Step 1: look-up tables:
    def fetchLookupTables(self):
        dict = compositionNetworksTagSets
        composition_tags, composition_tags_reverse = dict.dictionary()

        return composition_tags, composition_tags_reverse

    def fetchWordsAndPOSLookupTables(self):
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
        nn1_pos = load_pickles(list_of_lookup_tables[0])
        nn1_pos_reverse = load_pickles(list_of_lookup_tables[1])
        nn1_vocab = load_pickles(list_of_lookup_tables[2])
        nn1_vocab_reverse = load_pickles(list_of_lookup_tables[3])

        return nn1_vocab, nn1_pos, nn1_vocab_reverse, nn1_pos_reverse
        
    # Step 2: fetch dirty dataset
    def fetchDirtyDataset(self, dataset):
        ds = compositionNetworksDataset()
        dirty_sentences = ds.prepare_sentences(corpus_type=dataset)
        all_chunk_types = ['(NP', '(VP', '(PP','(SBAR','(ADJP','(PRT','(ADVP','(INTJ','(CONJP','(LST','(UCP','(PADDING','(UNAVAILABLE']

        all_phrases = []
        for type in all_chunk_types:
            for size in range(3,10,2):
                all_phrases.append(list(set(ds.identify_phrases_size_n(dirty_sentences, 
                                                                    chunk_type = type, 
                                                                    chunk_size=size)))) # NP phrases
        
        all_phrases_final = list(filter(None, all_phrases))
        return all_phrases_final

    # Step 3: Create tuples of (label, sentence) in a neat way:

    def cleanDirtyDataset(self,phrases):
        sentences_clean = []
        
        for phrase in phrases:
            
            sentence_pretty = []

            label = str(phrase[0]).replace('(','')
            
            sentence_ugly = list(phrase[1:])
            
            
            for item in sentence_ugly:
                token_less_ugly = str(item).replace('(','')
                token_less_ugly = token_less_ugly.replace(')','')
                token_less_ugly = token_less_ugly.replace('\n','')
                sentence_pretty.append(token_less_ugly)
            
            sentences_clean.append((label,sentence_pretty)) #tuple(label, sentence)

        return sentences_clean
            
    # Step 4: Convert tokens to idx:
    def prepareTensors(self,phrases):

        # pipelines
        def sample_word_pipeline(x):                        # word to idx
            return [self.penn_vocab[tok] for tok in x]
            
        def sample_pos_pipeline(x):                         # pos to idx
            return [self.penn_pos_tags[pos] for pos in x]



        samples = [item for sublist in phrases for item in sublist]
        all_samples = []
        skipped = 0
        for idx, sample in enumerate(samples):

            if len(sample)>6:
                continue
            
            label = sample[0]

            sentence, pos_tags = [], []

            #split sentence and pos tags
            for i in range(0,len(sample),2):
                try:
                    pos_tags.append(sample[1][i])
                    sentence.append(sample[1][i+1])
                except IndexError:
                    continue
                
            if len(sentence)<5:
                for i in range(0,5-len(sentence)):
                    sentence.append('PADDING')
                    pos_tags.append('PADDING')
            
            # token to idx conversion
            
            try:
                sentence_to_idx = sample_word_pipeline(sentence)
                pos_to_idx = sample_pos_pipeline(pos_tags)
                label_to_idx = self.composition_tags[label]
            except KeyError:
                skipped = skipped+1
                print(f"skipped {skipped} lines so far:")
                continue
            final_sample = [a + b for a, b in zip(sentence_to_idx, pos_to_idx)]

            sample_as_tensor = torch.tensor(final_sample)
            label_as_tensor = torch.tensor(label_to_idx)

            all_samples.append((label_as_tensor,sample_as_tensor))

        return all_samples

     
############################################################################################################

    def __init__(self, myDataset=None):

        self.myDataset = myDataset
        print(self.myDataset)

        # Step 1: look-up tables:
        print("Loading Penn Treebank look-up tables...")
        self.composition_tags, self.composition_tags_reverse = self.fetchLookupTables()
        self.penn_vocab, _, _, self.penn_pos_tags = self.fetchWordsAndPOSLookupTables()
        print(self.penn_pos_tags.keys())
        print("done!")

        # Step 2: Fetching Pre-processed dataset:
        print("Pre-processing Penn Treebank dataset...")
        self.cleanedRawDataset = self.fetchDirtyDataset(dataset=self.myDataset)
        self.prettyDataset = []
        for dataset in self.cleanedRawDataset:
            self.prettyDataset.append(self.cleanDirtyDataset(dataset))
        print("done!")

        # Step 3: Convert the dataset to tensors
        print("Transforming the dataset to tensors...")
        self.final_dataset = self.prepareTensors(self.prettyDataset)
        print("done!")

    def __len__(self):

        print(f"Total number of samples in this {self.myDataset} dataset: {len(self.final_dataset)}")

        return len(self.final_dataset)

    def __getitem__(self, idx) :

        return self.final_dataset[idx]
            

    
###################################################################

validation_dataset = DataLoader(dataset=composeDataset(myDataset="validation"),
                                batch_size=64,
                                shuffle=True)

for idx, sample in enumerate(validation_dataset):
    print(sample)
# print("Fetching Penn Treebank look-up tables...")

# fetch = composeDataset()
# composition_tags, composition_tags_reverse = fetch.fetchLookupTables()
# penn_vocab, penn_pos_tags, penn_vocab_reverse, penn_pos_tags_reverse = fetch.fetchWordsAndPOSLookupTables()
# print("done!")
# print("Fetching cleaned-raw dataset...")
# cleanedDataset = fetch.fetchDirtyDataset(dataset="validation")
# print("done!")
# prettyDataset = []
# for dataset in cleanedDataset:
#     prettyDataset.append(fetch.cleanDirtyDataset(dataset))

# final_dataset_text = fetch.prepareTensors(prettyDataset)
# print(len(final_dataset_text))
# print(final_dataset_text[:10])
# print('-'*100)
