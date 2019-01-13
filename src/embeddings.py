# Load the Embeddings

import numpy as np


def load_embeddings(emb_path, emb_dims):
    '''
    Load the embeddings from a text file
    
        :param emb_path: Path of the text file
        :param emb_dims: Embedding dimensions
        
        :return emb_tensor: tensor containing all word embeedings
        :return word_to_indx: dictionary with word:index
    '''

    # Load the file
    lines = open(emb_path).readlines()
    
    # Creating the list and adding the PADDING embedding
    emb_tensor = [np.zeros(emb_dims)]
    word_to_indx = {'PADDING_WORD':0}
    
    # For each line, save the word, the embedding and the word:index
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        
        if not len(emb) == emb_dims:
            continue
        
        # Update the embedding list and the word:index dictionary
        emb_tensor.append(list(np.float_(emb))) #[float(x) for x in emb])
        word_to_indx[word] = indx + 1
    
    # Turning the list into a numpy object
    emb_tensor = np.array(emb_tensor, dtype=np.float32)
    return emb_tensor, word_to_indx