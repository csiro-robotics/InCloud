import pickle 
import numpy as np 
from tqdm import tqdm 
from eval.eval_utils import get_latent_vectors
from sklearn.neighbors import KDTree

def eval_multisession(model, database_sets, query_sets):
    recall = np.zeros(25)
    count = 0
    similarity = [] 
    all_correct = []
    all_incorrect = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    database_sets = pickle.load(open(database_sets, 'rb'))
    query_sets = pickle.load(open(query_sets, 'rb'))

    for run in tqdm(database_sets, disable=False, desc = 'Getting database embeddings'):
        database_embeddings.append(get_latent_vectors(model, run))

    for run in tqdm(query_sets, disable=False, desc = 'Getting query embeddings'):
        query_embeddings.append(get_latent_vectors(model, run))

    for i in tqdm(range(len(query_sets)), desc = 'Getting Recall'):
        for j in range(len(query_sets)):
            if i == j:
                continue 
            pair_recall, pair_similarity, pair_opr, correct, incorrect = get_recall(
                i, j, database_embeddings, query_embeddings, query_sets, database_sets
            )
            recall += np.array(pair_recall)
            count += 1 
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)
            for x in correct:
                all_correct.append(x)
            for x in incorrect:
                all_incorrect.append(x)
    
    ave_recall = recall / count
    ave_recall_1 = ave_recall[0]
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'Recall@1%': ave_one_percent_recall, 'Recall@1': ave_recall_1, 'Recall@N': ave_recall}
    return stats


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code

    database_output = database_vectors[m]
    queries_output = query_vectors[n]


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    correct = []
    incorrect = []

    num_evaluated = 0

    for i in range(len(queries_output)): #size 
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                    correct.append(similarity)
                recall[j] += 1
                break
            else:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    incorrect.append(similarity)

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    
    return recall, top1_similarity_score, one_percent_recall, correct, incorrect

# TODO Write code to evaluate an individual run 