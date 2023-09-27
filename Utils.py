import numpy
import sklearn.cluster

def get_distance_cluster(
    pixel_value: float, 
    cluster: numpy.ndarray
) -> numpy.ndarray:
    return numpy.min(numpy.abs(pixel_value-cluster))
    
def probability_clusters(
    pixel_value: float, 
    clusters: numpy.ndarray
) -> numpy.ndarray:

    distances_clusters = numpy.array(
        [
            get_distance_cluster(pixel_value, cluster)
            for cluster in clusters
        ]
    )

    total = distances_clusters.sum()

    proba_not_in = numpy.array(
        [
            dist_cluster / total 
            for dist_cluster in distances_clusters 
        ]
    )

    proba_in = 1 - proba_not_in

    return proba_in


def probability_map(
    img: numpy.ndarray, 
    clusters: numpy.ndarray
) -> numpy.ndarray:
    
    n, m = img.shape
    nb_classes = clusters.shape[0]
    img_prob = numpy.zeros(shape=(nb_classes, n, m))

    for i in range(0, n):
        for j in range(0, m):
            img_prob[:, i, j] = probability_clusters(
                pixel_value = img[i, j],
                clusters = clusters
            )

    return img_prob

# def likely_probable(
#     proba_map: numpy.ndarray
# ) -> numpy.ndarray:
    
#     nb_classes, n, m = proba_map.shape
#     idx_lp = numpy.zeros_like(proba_map)

#     for i in range(0, n):
#         for j in range(0, m):
#             idx_lp[:, i, j] = numpy.where(
#                 proba_map[:, i, j] == proba_map[:, i, j].max()
#             )

#     return idx_lp

def thresholding_kmeans(img: numpy.ndarray, k: int) -> numpy.ndarray:

    img_vectorized = numpy.reshape(img, newshape=(-1, 1), order='F')
    _, idx, _ = sklearn.cluster.k_means(
        img_vectorized, 
        n_clusters=k, 
        init='k-means++', 
        random_state=0, 
        n_init='auto'
    )
    idx = numpy.reshape(idx, newshape=img.shape, order='F')

    cluster_mean = numpy.zeros(shape=k)

    for i in range(0, k):
        cluster_mean[i] = numpy.mean(img[idx==i])

    cluster_mean_sorted = numpy.sort(cluster_mean)

    thresholds = numpy.zeros(shape=k-1)

    for i in range(0, k-1):
        thresholds[i] = numpy.mean(cluster_mean_sorted[i:i+2])

    return thresholds


def create_probability_map(img: numpy.ndarray, k: int) -> numpy.ndarray:
    """Tenplate subtitute
    """

    img_vectorized = numpy.reshape(img, newshape=(-1, 1), order='F')
    _, idx, _ = sklearn.cluster.k_means(
        img_vectorized, 
        n_clusters=k, 
        init='k-means++', 
        random_state=0, 
        n_init='auto'
    )

    idx = numpy.reshape(idx, newshape=img.shape, order='F')

    # clusters = numpy.empty(shape=(k), dtype=numpy.ndarray)

    clusters = numpy.array(
        object = [ img[idx==i] for i in range(0, k) ],
        dtype=numpy.ndarray
    )
    
    proba_map = probability_map(
        img = img,
        clusters=clusters
    )

    return proba_map


def seg_result(
    ms_res: numpy.ndarray,
    thresholds: numpy.ndarray,
    k: int 
) -> numpy.ndarray:

    seg = numpy.zeros_like(ms_res)

    for i in range(0, k-1):

        if i == 0:
            temp = ms_res < thresholds[i]
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            if k == 2:
                temp = thresholds[i] <= ms_res
                seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        elif i == k-1:
            temp = (thresholds[i] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
            temp = (thresholds[i] <= ms_res)
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)
        else:
            temp = (thresholds[i-1] <= ms_res) \
                * (ms_res < thresholds[i])
            seg += temp * numpy.sum(temp*ms_res) / numpy.sum(temp)

    return seg