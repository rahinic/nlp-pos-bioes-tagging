class compositionNetworksTagSets():

    def dictionary():
        # compose dictionary of possible sentence composition tags
        composition_tags = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'ADVP', 'PRT', 'CONJP', 'INTJ', 'LST', 'UCP', 'PADDING', 'UNAVAILABLE']
        composition_tags_lkp, composition_tags_lkp_rev = {}, {}
        
        for idx, tag in enumerate(composition_tags):
            composition_tags_lkp[tag] = idx
            composition_tags_lkp_rev[idx] = tag
        
        print("Dictionary ready!")
        return composition_tags_lkp, composition_tags_lkp_rev

class compositionNetworksDataset():
    # fetch datasets
    def load_corpus(self,filename):
        filepath = "data/corpus/"
        f = open(filepath+str(filename),"r")
        content = f.readlines()

        return content

    def fetch_corpus(self,corpus_type):

        print(f"loading the {corpus_type} dataset")
        import sys

        types = ['train','test','validation']
        try:
            if corpus_type not in types:
                raise ValueError
        except ValueError as error:
            print('Invalid dataset type: possible values: "train/test/validation"')
            sys.exit(1)

        if corpus_type == "train":
            dataset = self.load_corpus(filename="02-21.10way.clean.txt")
        elif corpus_type == "test":
            dataset = self.load_corpus(filename="23.auto.clean.txt")
        elif corpus_type == "validation":
            dataset = self.load_corpus(filename="22.auto.clean.txt")

        return dataset

    def prepare_sentences(self,corpus_type):

        corpus_dataset = self.fetch_corpus(corpus_type)
        
        

        #import dict
        dict, dict_rev = compositionNetworksTagSets.dictionary()
        list_of_tags = list(dict.keys())

        return corpus_dataset


    def identify_phrases_size_n(self, find_chunks_from, chunk_type, chunk_size):

    #collects NP phrases of size 2 and 1 (at the end of sentence)
        list_of_phrases = []
        n = chunk_size 
        for sentence in find_chunks_from: # iterate through all sentences
            curr_sentence = sentence.split(' ')
            
            for idx, item in enumerate(curr_sentence): 

                # look if NP phrase exist in this sentence            
                if item == chunk_type:
                    NP_item_position = idx
                    if '))' in curr_sentence[idx:idx+n][-1]:
                        curr_phrase = curr_sentence[idx:idx+n] # n= 5
                        
                    else: continue #skipping chunks longer than 2 words
                    
                    # remove the end of sentence indicator:
                    if curr_phrase[-1] == '.)))\n':
                        curr_phrase =  curr_phrase[:-2]

                    list_of_phrases.append(tuple(curr_phrase))             
                
                # elif item != chunk_type: # skip until the next NP phrase in the same sentence
                #     continue
        return list_of_phrases

ds = compositionNetworksDataset()
dirty_sentences = ds.prepare_sentences(corpus_type="validation")
all_chunk_types = ['(NP', '(VP', '(PP','(SBAR','(ADJP','(PRT','(ADVP','(INTJ','(CONJP','(LST','(UCP','(PADDING','(UNAVAILABLE']

all_phrases = []
for type in all_chunk_types:
    for size in range(3,10,2):
        all_phrases.append(list(set(ds.identify_phrases_size_n(dirty_sentences, 
                                                            chunk_type = type, 
                                                            chunk_size=size)))) # NP phrases
  
all_phrases_final = list(filter(None, all_phrases))
# print(len(all_phrases_final))
# print(all_phrases_final[-1][:5])

# sentences = ds.fetch_corpus(corpus_type="validation")
# print(sentences[5])

        

