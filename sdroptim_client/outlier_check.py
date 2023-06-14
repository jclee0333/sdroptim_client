import os
import sys
import multiprocessing as mp
import pandas as pd
import gc
import numpy as np
from numpy import percentile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.use('Agg')
from sklearn.manifold import TSNE

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

############## PYOD models we use (pre-defined)
from pyod.models.abod import ABOD
#from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

import missingno as msno

##############

def data_loader_2d(csv_fullpath):
    try:
        ori_dataset = pd.read_csv(csv_fullpath, keep_default_na=True)
    except:
        ori_dataset = pd.read_csv(csv_fullpath, keep_default_na=True, encoding="ISO-8859-1")
    print(csv_fullpath, " loaded successfully.")
    #data cleaning
    dataset = pd.get_dummies(ori_dataset)
    dataset.replace(np.nan,"0",inplace=True)
    print(csv_fullpath, " has been cleaned.")
    #
    tsne = TSNE(n_components=2, n_jobs=8, n_iter=300, n_iter_without_progress=30) # should be modified due to interactive process (too slow)
    dataset_2d = tsne.fit_transform(dataset)
    print(csv_fullpath, " has been converted 2-dim. datasets for visualization by TSNE.")
    return dataset_2d, ori_dataset

#    if dataset.shape[1]>=2:
#        tsne = TSNE(n_components=2, n_jobs=-1) # should be modified due to interactive process
#        dataset_2d = tsne.fit_transform(dataset)
#        return dataset_2d, ori_dataset
#    else:
#        return dataset.values, ori_dataset

def get_outlier_prediction(all_parts):
    res = []
    for each in all_parts:
        i, clf_name, clf, X, outliers_fraction = each[0],each[1],each[2], each[3], each[4]
        #X = dataset_df
        xx, yy = np.meshgrid(np.linspace(X.min(),X.max()), np.linspace(X.min(), X.max()))
        print(i + 1, 'fitting', clf_name)
        # fit the data and tag outliers
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
        #results.append(np.where(y_pred==1)[0])
        outlier_index = np.where(y_pred==1)[0]
        threshold = percentile(scores_pred, 100 * outliers_fraction)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)
        res.append([clf_name, i, xx, yy, X, Z, threshold, y_pred, outlier_index])
        print(clf_name, "finished.")
    #return res
    if len(all_parts)==1:
        return res[0]
    else:
        return res

def plotting(chart_items, csv_file_name):
    plt.figure(figsize=(25, 20))
    for each in chart_items:
        clf_name, i, xx, yy, X, Z, threshold, y_pred, outlier_index  = each[0], each[1], each[2], each[3], each[4], each[5], each[6], each[7], each[8]
        subplot = plt.subplot(3, 4, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        #print(y_pred)
        inliers = subplot.scatter(X[np.where(y_pred==0)[0], 0], X[np.where(y_pred==0)[0], 1], c='white',s=10, edgecolor='k')
        outliers = subplot.scatter(X[np.where(y_pred==1)[0], 0], X[np.where(y_pred==1)[0], 1], c='black',s=20, edgecolor='k', marker='d')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], inliers, outliers],
            ['learned decision function', 'inliers', 'outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s n_outliers: %d" % (i + 1, clf_name, len(np.where(y_pred==1)[0])))
    outputpath = os.path.basename(csv_file_name).split('.')[0]+'_outliers.png'
    plt.savefig(outputpath)
    os.chmod(outputpath, 0o776)
    
def parallelizize_detecting_outlier_by_pool(all_parts, func, n_cores='auto'):
    if n_cores == 'auto':
        n_cores = min(int(mp.cpu_count() / 2), int(len(all_parts)))
    all_parts_split = np.array_split(all_parts, n_cores)
    pool = mp.Pool(n_cores)
    res = pool.map(func, all_parts_split)
    pool.close()
    pool.join()
    #
    ### 20230614 parallelization bug fix by jclee
    if len(all_parts) != len(res):
        res = [data for inner_list in res for data in inner_list]
    return res

def get_voted_index_to_remove(results, frequency_thres_for_n_vote = 0.5):
    from collections import Counter
    results_list=[]
    for each in results:
        results_list.append(list(each[8]))
    def flatten(input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    results_list_1d=flatten(results_list)
    results_count=Counter(results_list_1d)
    n_vote_thres = len(results) * frequency_thres_for_n_vote
    sorted_results_count = results_count.most_common()
    voted_index_to_remove_because_it_is_outlier = []
    for i, each in enumerate(sorted_results_count):
        if each[1]<n_vote_thres:
            break
        else:
            voted_index_to_remove_because_it_is_outlier.append(each[0])
    return voted_index_to_remove_because_it_is_outlier
         

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_name', help="name of targetCSVfile", default="")
    parser.add_argument('--outliers_fraction', help="default outliers_fraction", default=0.05)
    parser.add_argument('--voting_thres', help="voting threshold", default=0.5)
    args = parser.parse_args()
    outliers_fraction = args.outliers_fraction
    #####################
    random_state = np.random.RandomState(42)
    #detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
    #                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
    #                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
    #                 LOF(n_neighbors=50)]
    # Define nine outlier detection tools to be compared
    classifiers = {
        'Angle-based Outlier Detector (ABOD)':
            ABOD(contamination=outliers_fraction), ####
        #'Cluster-based Local Outlier Factor (CBLOF)':
        #    CBLOF(contamination=outliers_fraction,
        #          check_estimator=False, random_state=random_state),
        'Copula Based Outlier Detection (COPOD)': 
        COPOD(contamination=outliers_fraction), ####
        #'Feature Bagging':### --> too slow
        #    FeatureBagging(LOF(n_neighbors=35),
        #                   contamination=outliers_fraction,
        #                   random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS( ####
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state), ####
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction), ###
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction), ####
        'Local Outlier Factor (LOF)':
            LOF(n_neighbors=35, contamination=outliers_fraction), ####
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state), ####
        #'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction), ### --> too slow
        #'Principal Component Analysis (PCA)': PCA(
        #    contamination=outliers_fraction, random_state=random_state),### --> too slow
        #'Locally Selective Combination (LSCP)': LSCP(
        #    detector_list, contamination=outliers_fraction,### --> too slow
        #    random_state=random_state)
    }
    #################
    if os.path.exists(args.csv_file_name) == True:
        X, ori_dataset = data_loader_2d(args.csv_file_name)
        ###
        outputpath=os.path.basename(args.csv_file_name).split('.')[0]+"_missingplot_matrix.png"
        missing_bar = msno.matrix(ori_dataset.sample(100, replace=True))
        missing_bar.get_figure().savefig(outputpath)#, bbox_inches='tight')
        os.chmod(outputpath, 0o776)
        ###
        outputpath=os.path.basename(args.csv_file_name).split('.')[0]+"_missingplot_bar.png"
        missing_bar = msno.bar(ori_dataset)
        missing_bar.get_figure().savefig(outputpath)#, bbox_inches='tight')
        os.chmod(outputpath, 0o776)
        ###
        outputpath=os.path.basename(args.csv_file_name).split('.')[0]+"_missingplot_heatmap.png"
        missing_heatmap = msno.heatmap(ori_dataset, cmap="RdYlGn")
        missing_heatmap.get_figure().savefig(outputpath)#, bbox_inches='tight')
        os.chmod(outputpath, 0o776)
        print("Missing charts (matrix, bar, heatmap) has been generated.")
        ###
        classifiers_items = []
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            classifiers_items.append([i, clf_name, clf, X, outliers_fraction])
        results = parallelizize_detecting_outlier_by_pool(classifiers_items, get_outlier_prediction)
        plotting(results, args.csv_file_name)
        voted_index_to_remove_because_it_is_outlier = get_voted_index_to_remove(results, args.voting_thres)
        the_rest = list(set(pd.DataFrame(ori_dataset).index.tolist()) - set(voted_index_to_remove_because_it_is_outlier))
        ###
        outputpath=os.path.basename(args.csv_file_name).split('.')[0]+"_outliers_removed.csv"
        ori_dataset.iloc[the_rest].to_csv(outputpath, index=False)
        os.chmod(outputpath, 0o776)
        outputpath=os.path.basename(args.csv_file_name).split('.')[0]+"_outliers.csv"
        ori_dataset.iloc[voted_index_to_remove_because_it_is_outlier].to_csv(outputpath, index=False)
        os.chmod(outputpath, 0o776)
        print("Outlier detection process finished.")
        
