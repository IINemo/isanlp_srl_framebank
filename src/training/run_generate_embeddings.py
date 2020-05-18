from torch.utils.data import DataLoader
import fire
import pickle
from tqdm import tqdm
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
import numpy as np
import os


ELMO_PATH = 'http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz'


def main(feature_path, output_dir, model_url=ELMO_PATH, 
         cuda_device=0, batch_size=32):
    print('Loading ELMo model...')
    
    elmo = ELMoEmbedder(model_url, 
                        elmo_output_names=['elmo'], 
                        cuda_device=cuda_device)
    
    print('Done.')
    
    print('Loading features...')
    with open(feature_path, 'rb') as f:
        X_orig = pickle.load(f)
        
    print('Done.')
        
    X_orig = X_orig[['tokens', 'prd_address', 'arg_address']]
    
    embedded_verbs = []
    embedded_args  = []

    print('Embedding...')
    loader = DataLoader(list(range(X_orig.shape[0])), batch_size=batch_size, collate_fn=lambda _:_)
    #loader = DataLoader(list(range(1000)), batch_size=batch_size, collate_fn=lambda _:_)
    for batch in tqdm(loader):
        b_tokens = X_orig.tokens.iloc[batch].tolist()
        b_embed = elmo(b_tokens)

        for num, i in enumerate(batch):
            obj = X_orig.iloc[i]
            embedded_verbs.append(b_embed[num][min(obj.prd_address, len(obj.tokens) - 1)])
            embedded_args.append(b_embed[num][min(obj.arg_address, len(obj.tokens) - 1)])
    
    e_verbs = np.stack(embedded_verbs)
    e_args  = np.stack(embedded_args)
    
    print('Done.')
    
    print('Saving results...')
    np.save(os.path.join(output_dir, 'elmo_verbs_whole.npy'), e_verbs)
    np.save(os.path.join(output_dir, 'elmo_args_whole.npy'), e_args)
    print('Done.')
    
    
if __name__ == "__main__":
    fire.Fire(main)
    