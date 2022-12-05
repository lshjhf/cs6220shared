"""
    ##############################################
    Recommendations (``examples.recommendations``)
    ##############################################

    In this examples of collaborative filtering we consider movie recommendation using common MovieLens data set. It 
    represents typical cold start problem. A recommender system compares the user's profile to reference
    characteristics from the user's social environment. In the collaborative filtering approach, the recommender
    system identify users who share the same preference with the active user and propose items which the like-minded
    users favoured (and the active user has not yet seen).     

    We used the MovieLens 100k data set in this example. This data set consists of 100 000 ratings (1-5) from 943
    users on 1682 movies. Each user has rated at least 20 movies. Simple demographic info for the users is included. 
    Factorization is performed on a split data set as provided by the collector of the data. The data is split into 
    two disjoint sets each consisting of training set and a test set with exactly 10 ratings per user. 

    It is common that matrices in the field of recommendation systems are very sparse (ordinary user rates only a small
    fraction of items from the large items' set), therefore ``scipy.sparse`` matrix formats are used in this example. 

    The configuration of this example is SNMF/R factorization method using Random Vcol algorithm for initialization. 

    .. note:: MovieLens movies' rating data set used in this example is not included in the `datasets` and need to be
      downloaded. Download links are listed in the ``datasets``. Download compressed version of the MovieLens 100k. 
      To run the example, the extracted data set must exist in the ``MovieLens`` directory under ``datasets``. 

    .. note:: No additional knowledge in terms of ratings' timestamps, information about items and their
       genres or demographic information about users is used in this example. 

    To run the example simply type::

        python recommendations.py

    or call the module's function::

        import nimfa.examples
        nimfa.examples.recommendations.run()

    .. note:: This example uses ``matplotlib`` library for producing visual interpretation of the RMSE error measure. 

"""

from os.path import dirname, abspath
from os.path import join
from warnings import warn
import csv
import heapq

import numpy as np

import nimfa
from time import time
tic = time()

try:
    import matplotlib.pylab as plb
except ImportError as exc:
    warn("Matplotlib must be installed to run Recommendations example.")


def run():
    """
    Run SNMF/R on the MovieLens data set.

    Factorization is run on `ua.base`, `ua.test` and `ub.base`, `ub.test` data set. This is MovieLens's data set split 
    of the data into training and test set. Both test data sets are disjoint and with exactly 10 ratings per user
    in the test set. 
    """

    for data_set in ['ua', 'ub']:
        V = read(data_set)
        iters = [i for i in range(1, 100, 10)]
        ranks = [i for i in range(2, 50, 5)]
        beta = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5]
        # for i in range(len(iters)):
        #     W, H = factorize(V, 30, iters[i])
        #     print(rmse(W, H, data_set))
        # for i in range(len(ranks)):
        #     W, H = factorize(V, ranks[i], 20)
        #     print(rmse(W, H, data_set))
        for i in beta:
            W, H = factorize(V, 30, 20, i)
            rmse(W, H, data_set)


def factorize(V, rank, iter, beta):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 

    Return basis and mixture matrices of the fitted factorization model. 

    :param V: The MovieLens data matrix. 
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank, max_iter=iter, version='r', eta=1.,
                      beta=beta, i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                            fit.distance(metric='euclidean'),
                                                            sparse_w, sparse_h))
    return fit.basis(), fit.coef()


def read(data_set):
    """
    Read movies' ratings data from MovieLens data set. 

    :param data_set: Name of the split data set to be read.
    :type data_set: `str`
    """
    print("Read MovieLens data set")
    fname = join(dirname(dirname(abspath(__file__))), "datasets", "MovieLens", "%s.base" % data_set)
    fname = "u1.base"
    V = np.ones((6040, 3952)) * 2.5

    for line in open(fname):
        u, i, r, _ = list(map(int, line.split()))
        V[u - 1, i - 1] = r
    return V


def rmse(W, H, data_set):
    """
    Compute the RMSE error rate on MovieLens data set.

    :param W: Basis matrix of the fitted factorization model.
    :type W: `numpy.matrix`
    :param H: Mixture matrix of the fitted factorization model.
    :type H: `numpy.matrix`
    :param data_set: Name of the split data set to be read. 
    :type data_set: `str`
    """
    fname = join(dirname(dirname(abspath(__file__))), "datasets", "MovieLens", "%s.test" % data_set)
    fname = "u1.test"
    rmse = []
    mae = []
    result = np.ones((6040, 3952)) * 6
    result2 = np.ones((6040, 3952)) * 40
    actual = np.ones((6040, 3952)) * 6
    worst = []
    best = []
    result3 = np.ones((6040, 3952)) * 40
    # store_low = []
    # store_high = []
    # info_low = []
    # info_high = []
    for line in open(fname):
        u, i, r, _ = list(map(int, line.split()))
        actual[u-1][i-1] = r
        sc = max(min((W[u - 1, :] * H[:, i - 1])[0, 0], 5), 1) # in range 1 to 5 for one user and movie: u-1*i-1  ...  |
        rmse.append((sc - r) ** 2 )
        mae.append(abs(sc - r))
        result[u-1][i-1] = abs(sc - r)
        result2[u-1][i-1] = abs((sc - r) ** 2)
        result3[u - 1][i - 1] = sc
    #     if len(worst) < 5:
    #         heapq.heappush(worst, ((sc - r) ** 2, r, sc, u, i))
    #     else:
    #         c = heapq.heappop(worst)
    #         if (sc - r) ** 2 > c[0]:
    #             heapq.heappush(worst, ((sc - r) ** 2, r, sc, u, i))
    #         else:
    #             heapq.heappush(worst, c)
    #
    #     if len(best) < 5:
    #         heapq.heappush(best, (-(sc - r) ** 2, r, sc, u, i))
    #     else:
    #         c = heapq.heappop(best)
    #         if -(sc - r) ** 2 > c[0]:
    #             heapq.heappush(best, (-(sc - r) ** 2, r, sc, u, i))
    #         else:
    #             heapq.heappush(best, c)
    #
    #
    #
    # print(best)
    # print(worst)

    print("RMSE: %5.3f" % np.sqrt(np.mean(rmse)))
    print("MAE: " + str(np.mean(mae)))
    # np.savetxt('rating_result.csv', result3, delimiter=',')
    return np.sqrt(np.mean(rmse))
    # np.savetxt('my_result_new.csv', result, delimiter=',')
    # np.savetxt('my_result_new_rmse.csv', result2, delimiter=',')
    # test_ids = [3845, 728, 3660, 5207, 1939, 380, 231, 18, 621, 596, 424, 339, 907, 4073, 3730, 5214, 3598, 2744]
    # for item in test_ids:
    #     print(str(item) + str(sum(result2[item-1])/3952))



if __name__ == "__main__":
    """Run the Recommendations example."""
    run()
    print('data read in', time() - tic, 'seconds')
