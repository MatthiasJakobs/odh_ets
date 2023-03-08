import torch
import numpy as np

from tsx.distances import euclidean
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tsx.utils import to_random_state

# For each entry in rocs, return the closest to x using dist_fn
def find_closest_rocs(x, rocs, dist_fn=euclidean):
    closest_rocs = []
    closest_models = []

    for model in range(len(rocs)):
        rs = rocs[model]
        distances = [dist_fn(x.squeeze(), r.squeeze()) for r in rs]
        if len(distances) != 0:
            closest_rocs.append(rs[np.argsort(distances)[0]])
            closest_models.append(model)
    return closest_models, closest_rocs

def cluster_rocs(best_models, clostest_rocs, nr_desired_clusters, dist_fn=euclidean, random_state=None):
    rng = to_random_state(random_state)
    if nr_desired_clusters == 1:
        return best_models, clostest_rocs

    new_closest_rocs = []

    # Cluster into the desired number of left-over models.
    tslearn_formatted = to_time_series_dataset(clostest_rocs)
    km = TimeSeriesKMeans(n_clusters=nr_desired_clusters, metric='euclidean', random_state=rng)
    C = km.fit_predict(tslearn_formatted)
    C_count = np.bincount(C)

    # Final model selection
    G = []

    for p in range(len(C_count)):
        # Under all cluster members, find the one maximizing distance to current point
        cluster_member_indices = np.where(C == p)[0]
        # Since the best_models (and closest_rocs) are sorted by distance to x (ascending), 
        # choosing the first one will always minimize distance
        if len(cluster_member_indices) > 0:
            #idx = cluster_member_indices[-1]
            idx = cluster_member_indices[0]
            G.append(best_models[idx])
            new_closest_rocs.append(clostest_rocs[idx])

    return G, new_closest_rocs

def select_topm(models, rocs, x, upper_bound, dist_fn=euclidean):
    # Select top-m until their distance is outside of the upper bounds
    topm_models = []
    topm_rocs = []
    distances_to_x = np.zeros((len(rocs)))
    for idx, r in enumerate(rocs):
        distance_to_x = dist_fn(r.squeeze(), x.squeeze())
        distances_to_x[idx] = distance_to_x

        if distance_to_x <= upper_bound:
            topm_models.append(models[idx])
            topm_rocs.append(r)

    return topm_models, topm_rocs

