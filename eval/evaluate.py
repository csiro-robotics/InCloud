from torchpack.utils.config import configs 
import numpy as np 

import argparse 
import torch
from models.model_factory import model_factory
from eval.eval_multisession import eval_multisession
from eval.eval_singlesession import eval_singlesession
import pandas as pd 

def evaluate(model, env_idx):
    # Wrapper of other eval functions for incremental learning
    stats = {}
    for env in configs.eval.environments.keys():
        if configs.eval.environments[env]['stage_introduced'] <= env_idx or env_idx == -1: # Only eval on visited environments
            database_files = configs.eval.environments[env]['database_files']
            query_files = configs.eval.environments[env]['query_files']
            env_recall_1 = []
            for d, q in zip(database_files, query_files):
                if q != None:
                    env_recall_1.append(eval_multisession(model, d, q)['Recall@1'])
                else:
                    world_thresh = configs.eval.world_thresh[env]
                    false_pos_thresh = configs.eval.false_pos_thresh[env]
                    time_thresh = configs.eval.time_thresh[env]
                    env_recall_1.append(eval_singlesession(model, d, world_thresh, false_pos_thresh, time_thresh)['Recall@1'])
            env_recall_1 = np.mean(env_recall_1)
            stats[env] = env_recall_1
    print(stats)
    return stats 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True)
    parser.add_argument('--ckpt', type = str, required = True)
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    print(configs)

    model = model_factory(ckpt = torch.load(args.ckpt))
    stats = evaluate(model, -1)
    final = pd.DataFrame(columns = ['Recall@1'])
    for k in stats:
        final.loc[k] = [stats[k]]
    final.loc['Average'] = final.mean(0)
    print(stats)
    print(final)