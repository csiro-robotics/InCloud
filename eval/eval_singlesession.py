import torch 
import pickle 
import numpy as np 
from tqdm import tqdm 
from eval.eval_utils import get_latent_vectors
from torchpack.utils.config import configs 


def euclidean_distance(query, database):
    return torch.cdist(torch.tensor(query).unsqueeze(0).unsqueeze(0), torch.tensor(database).unsqueeze(0)).squeeze().numpy()

def cosine_dist(query, database):
    return np.array(1 - torch.einsum('D,ND->N', torch.tensor(query), torch.tensor(database)))

def eval_singlesession(model, database, world_thresh, false_pos_thresh, time_thresh):
    # Get embeddings, timestamps,coords and start time 
    database_dict = pickle.load(open(database, 'rb'))
    embeddings = get_latent_vectors(model, database_dict) # N x D, in chronological order
    timestamps = [database_dict[k]['timestamp'] for k in range(len(database_dict.keys()))]
    coords = np.array([[database_dict[k]['easting'],database_dict[k]['northing']] for k in range(len(database_dict.keys()))])
    start_time = timestamps[0]



    # Thresholds, other trackers
    thresholds = np.linspace(configs.eval.thresh_min, configs.eval.thresh_max, configs.eval.num_thresholds) # 0, 1, 1000 by default, TODO remove later
    num_thresholds = len(thresholds)

    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    # Get similarity function 
    if configs.eval.similarity == 'cosine':
        dist_func = cosine_dist
    elif configs.eval.similarity == 'euclidean':
        dist_func = euclidean_distance
    else:
        raise ValueError(f'No supported distance function for {configs.eval.similarity}')
        
    num_revisits = 0
    num_correct_loc = 0

    for query_idx in tqdm(range(len(database_dict)), desc = 'Evaluating Embeddings'):

        query_embedding = embeddings[query_idx]
        query_timestamp = timestamps[query_idx]
        query_coord = coords[query_idx]

        # Sanity check time 
        if (query_timestamp - start_time - time_thresh) < 0:
            continue 
        
        # Build retrieval database 
        tt = next(x[0] for x in enumerate(timestamps) if x[1] > (query_timestamp - time_thresh))
        seen_embeddings = embeddings[:tt+1] # Seen x D
        seen_coords = coords[:tt+1] # Seen x 2

        # Get distances in feat space and real world
        dist_seen_embedding = dist_func(query_embedding, seen_embeddings)
        dist_seen_world = euclidean_distance(query_coord, seen_coords)

        # Check if re-visit
        if np.any(dist_seen_world < world_thresh):
            revisit = True 
            num_revisits += 1
        else:
            revisit = False 

        # Get top-1 candidate and distances in real world, embedding space 
        top1_idx = np.argmin(dist_seen_embedding)
        top1_embed_dist = dist_seen_embedding[top1_idx]
        top1_world_dist = dist_seen_world[top1_idx]

        if top1_world_dist < world_thresh:
            num_correct_loc += 1

        # Evaluate top-1 candidate 
        for thresh_idx in range(num_thresholds):
            threshold = thresholds[thresh_idx]

            if top1_embed_dist < threshold: # Positive Prediction
                if top1_world_dist < world_thresh:
                    num_true_positive[thresh_idx] += 1
                elif top1_world_dist > configs.eval.false_positive_thresh:
                    num_false_positive[thresh_idx] += 1
            else: # Negative Prediction
                if not revisit:
                    num_true_negative[thresh_idx] += 1
                else:
                    num_false_negative[thresh_idx] += 1

    # Find F1Max and Recall@1 
    recall_1 = num_correct_loc / num_revisits

    F1max = 0.0 
    for thresh_idx in range(num_thresholds):
        nTruePositive = num_true_positive[thresh_idx]
        nFalsePositive = num_false_positive[thresh_idx]
        nTrueNegative = num_true_negative[thresh_idx]
        nFalseNegative = num_false_negative[thresh_idx]

        nTotalTestPlaces = nTruePositive + nFalsePositive + nTrueNegative + nFalseNegative

        Precision = 0.0
        Recall = 0.0
        Prev_Recall = 0.0
        F1 = 0.0

        if nTruePositive > 0.0:
            Precision = nTruePositive / (nTruePositive + nFalsePositive)
            Recall = nTruePositive / (nTruePositive + nFalseNegative)
            F1 = 2 * Precision * Recall * (1/(Precision + Recall))

        if F1 > F1max:
            F1max = F1 
            thresh_max = thresholds[thresh_idx]

    # print(f'Num Revisits : {num_revisits}')
    # print(f'Num. Correct Locations : {num_correct_loc}')
    # print(f'Recall@1: {recall_1}')
    # print(f'F1max: {F1max}')

    return {'F1max': F1max, 'Recall@1': recall_1}
