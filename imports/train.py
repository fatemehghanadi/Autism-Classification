# Copyright (c) 2019 Mwiza Kunda
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import time
import pandas as pd
import numpy as np
from nilearn import connectome
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
import scipy.stats as sc
import scipy.io as sio
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif,SelectFdr, SelectFpr,chi2,f_regression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc
# from sklearn.svm import LinearSVC
from imports.utils import arg_parse
from imports import KHSIC as KHSIC
from imports import MIDA as MIDA
from imports import preprocess_data as reader
from .utils import root_dir_default, data_folder_name_default
from config import get_cfg_defaults

# root_folder = "D:/ML_data/brain/qc"
# data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')


# Transform test data using the transformer learned on the training data
def process_test_data(timeseries, transformer, ids, params, k, seed, validation_ext, save_path=None):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    test_data = conn_measure.fit_transform(timeseries)

    if params['connectivity'] == 'tangent':
        connectivity = transformer.transform(timeseries)
    else:
        connectivity = transformer.transform(test_data)

    if save_path is None:
        save_path = os.path.join(root_dir_default, data_folder_name_default)
    atlas_name = params['atlas']
    kind = params['connectivity']

    for i, subj_id in enumerate(ids):
        subject_file = os.path.join(save_path, subj_id, subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_'
                                    + str(k) + '_' + str(seed) + '_' + validation_ext + str(params['n_subs']) +
                                    '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity[i]})

    # Process timeseries for tangent train/test split


def process_timeseries(subject_ids, train_ind, test_ind, params, k, seed, validation_ext):
    atlas = params['atlas']
    kind = params['connectivity']
    data_path = params["data_path"]
    timeseries = reader.get_timeseries(subject_ids, atlas, silence=True, data_path=data_path)
    train_timeseries = [timeseries[i] for i in train_ind]
    subject_ids_train = [subject_ids[i] for i in train_ind]
    test_timeseries = [timeseries[i] for i in test_ind]
    subject_ids_test = [subject_ids[i] for i in test_ind]

    print('computing tangent connectivity features..')
    transformer = reader.subject_connectivity(train_timeseries, subject_ids_train, atlas, kind, k)
    test_data_save = process_test_data(test_timeseries, transformer, subject_ids_test, params, k, seed, validation_ext)


# Grid search CV 
def grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=None, domain_ft=None, label_info=None):
    # MIDA parameter search
    mu_vals = [0.5, 0.75, 1.0]
    h_vals = [50, 150, 300]

    # Add phenotypes or not
    add_phenotypes = params['phenotypes']

    # Algorithm choice
    algorithm = params['algorithm']

    # Model choice
    model = params['model']

    # Seed
    seed = params['seed']

    # best parameters and 5CV accuracy
    best_model = {}
    best_model['acc'] = 0

    # Grid search formulation
    if algorithm in ['LR', 'SVM']:
        C_vals = [1, 5, 10]
        if algorithm == 'LR':
            max_iter_vals = [100000]
            alg = LogisticRegression(random_state=seed, solver='lbfgs')
        else:
            max_iter_vals = [100000]
            alg = svm.SVC(random_state=seed, kernel='linear')
        parameters = {'C': C_vals, 'max_iter': max_iter_vals}
    else:
        alpha_vals = [0.25, 0.5, 0.75]
        parameters = {'alpha': alpha_vals}
        alg = RidgeClassifier(random_state=seed)

    if model in ['MIDA', 'SMIDA']:
        for mu in mu_vals:
            for h in h_vals:
                x_data = features
                x_data = MIDA.MIDA(x_data, domain_ft, mu=mu, h=h, y=None)
                if add_phenotypes:
                    x_data = np.concatenate([x_data, phenotype_ft], axis=1)
                clf = GridSearchCV(alg, parameters, cv=5)
                clf.fit(x_data[train_ind], y[train_ind].ravel())
                if clf.best_score_ > best_model['acc']:
                    best_model['mu'] = mu
                    best_model['h'] = h
                    best_model = dict(best_model, **clf.best_params_)
                    best_model['acc'] = clf.best_score_

    else:
        x_data = features
        if add_phenotypes:
            x_data = np.concatenate([x_data, phenotype_ft], axis=1)
        clf = GridSearchCV(alg, parameters, cv=5)
        clf.fit(x_data[train_ind], y[train_ind].ravel())
        if clf.best_score_ > best_model['acc']:
            best_model = dict(best_model, **clf.best_params_)
            best_model['acc'] = clf.best_score_

    return best_model


# Ensemble models with different FC measures 
def leave_one_site_out_ensemble(params, subject_ids, features, y_data, y, phenotype_ft, phenotype_raw):
    results_acc = []
    results_auc = []
    all_pred_acc = np.zeros(y.shape)
    all_pred_auc = np.zeros(y.shape)

    algorithm = params['algorithm']
    seed = params['seed']
    atlas = params['atlas']
    num_domains = params['n_domains']
    validation_ext = params['validation_ext']
    filename = params['filename']
    num_subjects = params["n_subs"]
    data_path = params["data_path"]
    connectivities = {0: 'correlation', 1: 'TPE', 2: 'TE'}
    features_c = reader.get_networks(subject_ids, kind='correlation', data_path=data_path, iter_no='', atlas=atlas)
    add_phenotypes = params['phenotypes']

    for i in range(num_domains):
        k = i
        train_ind = np.where(phenotype_raw[:, 1] != i)[0]
        test_ind = np.where(phenotype_raw[:, 1] == i)[0]

        # load tangent pearson features
        try:
            pass
        except:
            features_t = reader.get_networks(subject_ids, kind='TPE', data_path=data_path, iter_no=k, seed=seed,
                                             validation_ext=validation_ext, n_subjects=params['n_subs'],
                                             atlas=atlas)
            print("Tangent features not found. reloading timeseries data")
            time.sleep(10)
            params['connectivity'] = 'TPE'
            process_timeseries(subject_ids, train_ind, test_ind, params, k, seed, validation_ext)
            features_t = reader.get_networks(subject_ids, kind='TPE', data_path=data_path, iter_no=k, seed=seed,
                                             validation_ext=validation_ext, n_subjects=params['n_subs'],
                                             atlas=atlas)

        # load tangent timeseries features
        try:
            features_tt = reader.get_networks(subject_ids, kind='TE', data_path=data_path, iter_no=k, seed=seed,
                                              validation_ext=validation_ext, n_subjects=params['n_subs'],
                                              atlas=atlas)
        except:
            print("Tangent features not found. reloading timeseries data")
            time.sleep(10)
            params['connectivity'] = 'TE'
            process_timeseries(subject_ids, train_ind, test_ind, params, k, seed, validation_ext)
            features_tt = reader.get_networks(subject_ids, kind='TE', data_path=data_path, iter_no=k, seed=seed,
                                              validation_ext=validation_ext, n_subjects=params['n_subs'],
                                              atlas=atlas)

        # all loaded features
        features = [features_c, features_t, features_tt]

        all_best_models = []
        x_data_ft = []
        if params['model'] == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            for ft in range(3):
                best_model = grid_search(params, train_ind, test_ind, features[ft], y, y_data,
                                         phenotype_ft=phenotype_ft, domain_ft=domain_ft)
                print('for', connectivities[ft], ', best parameters from 5CV grid search are: \n', best_model)
                x_data = MIDA.MIDA(features[ft], domain_ft, mu=best_model['mu'], h=best_model['h'], y=None)
                best_model.pop('mu')
                best_model.pop('h')
                best_model.pop('acc')
                all_best_models.append(best_model)
                x_data_ft.append(x_data)

        else:
            for ft in range(3):
                best_model = grid_search(params, train_ind, test_ind, features[ft], y, y_data)
                print('best parameters from 5CV grid search are: \n', best_model)
                best_model.pop('acc')
                all_best_models.append(best_model)
                x_data_ft.append(features[ft])

        algs = []
        preds_binary = []
        preds_decision = []

        # fit and compute predictions from all three models
        for ft in range(3):
            if add_phenotypes == True:
                x_data = np.concatenate([x_data, phenotype_ft], axis=1)

            if algorithm == 'LR':
                clf = LogisticRegression(random_state=seed, solver='lbfgs', **all_best_models[ft])
            elif algorithm == 'SVM':
                clf = svm.SVC(kernel='linear', random_state=seed, **all_best_models[ft])
            else:
                clf = RidgeClassifier(random_state=seed, **all_best_models[ft])

            algs.append(clf.fit(x_data_ft[ft][train_ind], y[train_ind].ravel()))
            preds_binary.append(clf.predict(x_data_ft[ft][test_ind]))
            preds_decision.append(clf.decision_function(x_data_ft[ft][test_ind]))

        # mode prediciton
        mode_predictions = sc.mode(np.hstack([preds_binary[j][np.newaxis].T for j in range(3)]), axis=1)[0].ravel()
        all_pred_acc[test_ind, :] = mode_predictions[:, np.newaxis]

        # Compute the accuracy
        lin_acc = accuracy_score(y[test_ind].ravel(), mode_predictions)

        # mean decision score
        mean_predictions = np.hstack([preds_decision[j][:, np.newaxis] for j in range(3)]).mean(axis=1)
        all_pred_auc[test_ind, :] = mean_predictions[:, np.newaxis]

        # Compute the AUC
        lin_auc = roc_auc_score(y[test_ind], mean_predictions)

        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-" * 100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: " + str(lin_auc))
        print("-" * 100)
    avg_acc = np.array(results_acc).mean()
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    weighted_acc = (y == all_pred_acc).sum() / params['n_subs']
    weighted_auc = roc_auc_score(y, all_pred_auc)

    print("accuracy average", avg_acc)
    print("standard deviation accuracy", std_acc)
    print("auc average", avg_auc)
    print("standard deviation auc", std_auc)
    print("(weighted) accuracy", weighted_acc)
    print("(weighted) auc", weighted_auc)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')


# leave one site out application performance
def leave_one_site_out(params, subject_ids, features, y_data, y, phenotype_ft, phenotype_raw):
    results_acc = []
    results_auc = []
    all_pred_acc = np.zeros(y.shape)
    all_pred_auc = np.zeros(y.shape)

    algorithm = params['algorithm']
    seed = params['seed']
    connectivity = params['connectivity']
    atlas = params['atlas']
    num_domains = params['n_domains']
    validation_ext = params['validation_ext']
    filename = params['filename']
    add_phenotypes = params['phenotypes']
    num_subjects = params["n_subs"]
    data_path = params["data_path"]
    pheno_only = params["pheno_only"]
    for i in range(num_domains):
        k = i
        train_ind = np.where(phenotype_raw[:, 1] != i)[0]
        test_ind = np.where(phenotype_raw[:, 1] == i)[0]

        if pheno_only:
            best_model = grid_search(params, train_ind, test_ind, phenotype_ft, y, y_data, phenotype_ft=phenotype_ft)
            x_data = phenotype_ft
        else:
            if connectivity in ['TPE', 'tangent']:
                try:
                    features = reader.get_networks(subject_ids, kind=connectivity, data_path=data_path, iter_no=k,
                                                   seed=seed, validation_ext=validation_ext,
                                                   n_subjects=params['n_subs'], atlas=atlas)
                except:
                    print("Tangent features not found. reloading timeseries data")
                    time.sleep(10)
                    process_timeseries(subject_ids, train_ind, test_ind, params, k, seed, validation_ext)
                    features = reader.get_networks(subject_ids, kind=connectivity, data_path=data_path, iter_no=k,
                                                   seed=seed, validation_ext=validation_ext,
                                                   n_subjects=params['n_subs'], atlas=atlas)

            if params['model'] == 'MIDA':
                domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
                best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft,
                                         domain_ft=domain_ft)
                print('best parameters from 5CV grid search: \n', best_model)
                x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], y=None)
                best_model.pop('mu')
                best_model.pop('h')
            else:
                best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft)
                print('best parameters from 5CV grid search: \n', best_model)
                x_data = features

            if add_phenotypes:
                x_data = np.concatenate([x_data, phenotype_ft], axis=1)

        # Remove accuracy key from best model dictionary
        best_model.pop('acc')

        # Set classifier
        if algorithm == 'LR':
            clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)
        elif algorithm == 'SVM':
            clf = svm.SVC(random_state=seed, kernel='linear', **best_model)
        else:
            clf = RidgeClassifier(random_state=seed, **best_model)

        # Fit classifier
        clf.fit(x_data[train_ind, :], y[train_ind].ravel())

        # Compute the accuracy
        lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
        y_pred = clf.predict(x_data[test_ind, :])
        all_pred_acc[test_ind, :] = y_pred[:, np.newaxis]

        # Compute the AUC
        pred = clf.decision_function(x_data[test_ind, :])
        all_pred_auc[test_ind, :] = pred[:, np.newaxis]
        lin_auc = roc_auc_score(y[test_ind], pred)

        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-" * 100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: " + str(lin_auc))
        print("-" * 100)

    avg_acc = np.array(results_acc).mean()
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    weighted_acc = (y == all_pred_acc).sum() / params['n_subs']
    weighted_auc = roc_auc_score(y, all_pred_auc)

    print("(unweighted) accuracy average", avg_acc)
    print("(unweighted) standard deviation accuracy", std_acc)
    print("(unweighted) auc average", avg_auc)
    print("(unweighted) standard deviation auc", std_auc)
    print("(weighted) accuracy", weighted_acc)
    print("(weighted) auc", weighted_auc)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')


# 10 fold CV 
def train_10CV(params, subject_IDs, features, y_data, y, phenotype_ft, phenotype_raw):
    results_acc = []
    results_auc = []

    algorithm = params['algorithm']
    seed = params['seed']
    connectivity = params['connectivity']
    atlas = params['atlas']
    num_domains = params['n_domains']
    model = params['model']
    add_phenotypes = params['phenotypes']
    filename = params['filename']
    validation_ext = params['validation_ext']
    num_subjects = params["n_subs"]
    data_path = params["data_path"]
    pheno_only = params["pheno_only"]
    if seed == 123:
        skf = StratifiedKFold(n_splits=10)
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    all_fpr = []
    all_tpr = []
    all_auc = []
    for sets, k in zip(list(reversed(list(skf.split(np.zeros(num_subjects), np.squeeze(y))))), list(range(10))):
        train_ind = sets[0]
        test_ind = sets[1]

        if connectivity in ['TPE', 'tangent']:
            try:
                features = reader.get_networks(subject_IDs, kind=connectivity, data_path=data_path, iter_no=k,
                                               seed=seed, validation_ext=validation_ext,
                                               n_subjects=params['n_subs'], atlas=atlas)
            except:
                print("Tangent features not found. reloading timeseries data")
                time.sleep(10)
                process_timeseries(subject_IDs, train_ind, test_ind, params, k, seed, validation_ext)
                features = reader.get_networks(subject_IDs, kind=connectivity, data_path=data_path, iter_no=k,
                                               seed=seed, validation_ext=validation_ext,
                                               n_subjects=params['n_subs'], atlas=atlas)
        # PCA & MI --------------------------------
        args = arg_parse()

        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.cfg)
        cfg.freeze()

        pcaflag=cfg.METHOD.PCA
        MIflag=cfg.METHOD.ANOVA

        # print(params['n_subs'])
        if pcaflag:
            train_ind.sort()
            test_ind.sort()
            
            num_comp=cfg.METHOD.PCA_FEATURES
            pca = PCA(n_components=num_comp, random_state=seed)
            train_features_pca = pca.fit_transform(features[train_ind])
            test_features_pca = pca.transform(features[test_ind])
            box_plot_before=features
            num_mi=cfg.METHOD.ANOVA_FEATURES
            if MIflag:
                selector = SelectKBest(f_classif, k=num_mi)
                # selector = SelectKBest(score_func=f_regression, k=num_mi)
                selector.fit(train_features_pca, y[train_ind])

                train_mi = selector.transform(train_features_pca)
                test_mi = selector.transform(test_features_pca)
                print('MI Done.')

            features=[[0]*num_comp]*int(params['n_subs'])
            i=0
            for tr in train_ind:
                if MIflag:
                    features[tr] = train_mi[i]
                else:
                    features[tr] = train_features_pca[i]
                i=i+1
            j=0
            for ts in test_ind:
                if MIflag:
                    features[ts] = test_mi[j]
                else:
                    features[ts] = test_features_pca[j]
                j=j+1
            features=np.array(features)
            print('PCA Done.')
        # t-SNE  ------------------------------------------------------------
        # train_mi1=train_mi
        # print(train_mi1)
        # print(train_mi1.shape)
        # tsne = TSNE(n_components=2, random_state=42)
        # transformed_tsne = tsne.fit_transform(train_mi1)

        # # Plot results
        # plt.figure(figsize=(8, 6))

        # plt.scatter(transformed_tsne[:, 0], transformed_tsne[:, 1], c=y[train_ind], cmap='viridis')
        # plt.title('Data Representation After Employing ANOVA and PCA Techniques')
        # plt.colorbar()

        # # Save the plot as a PDF file
        # plt.savefig('./tsne_anova_pca.pdf')

        # tsne = TSNE(n_components=2, random_state=42)
        # transformed_tsne = tsne.fit_transform(features)

        # # Plot results
        # plt.figure(figsize=(8, 6))

        # plt.scatter(transformed_tsne[:, 0], transformed_tsne[:, 1], c=y, cmap='viridis')
        # plt.title('Data Representation Before Employing ANOVA and PCA Techniques')
        # plt.colorbar()

        # # Save the plot as a PDF file
        # plt.savefig('./tsne_anova_raw.pdf')
        # # 3D plot --------------------------------------------------------
            
        # Assuming train_mi1 is your data after employing ANOVA and PCA techniques
        # Assuming features is your original data before employing ANOVA and PCA techniques

        # Create a TSNE object for 3D visualization
        # tsne_3d = TSNE(n_components=3, random_state=42)

        # Fit and transform train_mi1 data for 3D visualization
        # transformed_tsne_3d = tsne_3d.fit_transform(train_mi1)

        # Plot results for train_mi1 data
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(transformed_tsne_3d[:, 0], transformed_tsne_3d[:, 1], transformed_tsne_3d[:, 2], c=y[train_ind], cmap='viridis')
        # ax.set_title('Data Representation After Employing ANOVA and PCA Techniques')
        # ax.set_xlabel('Feature 1')
        # ax.set_ylabel('Feature 2')
        # ax.set_zlabel('Feature 3')

        # Save the plot as a PDF file
        # plt.savefig('./tsne_anova_pca_3d.pdf')

        # # Fit and transform features data for 3D visualization
        # transformed_tsne_3d_features = tsne_3d.fit_transform(features)

        # # Plot results for features data
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(transformed_tsne_3d_features[:, 0], transformed_tsne_3d_features[:, 1], transformed_tsne_3d_features[:, 2], c=y, cmap='viridis')
        # ax.set_title('Data Representation Before Employing ANOVA and PCA Techniques')
        # ax.set_xlabel('Feature 1')
        # ax.set_ylabel('Feature 2')
        # ax.set_zlabel('Feature 3')

        # # Save the plot as a PDF file
        # plt.savefig('./tsne_anova_raw_3d.pdf')
        
        # plt.show()
        # Box plot -------------------------------------
        #Extract the most discriminative features from train_mi1
        # num_features = 5
        # selector = SelectKBest(f_classif, k=num_features)
        # selector.fit(train_mi, y[train_ind])
        # best_20_features = selector.transform(train_mi)
                
        
        # # Split the data based on class labels
        # c = y[train_ind]
        # print(best_20_features)
        # print(c)
        
        # indices_1 = [i for i, val in enumerate(c) if val == 1]
        # indices_2 = [i for i, val in enumerate(c) if val == 2]

        # class_0_data = best_20_features[indices_1]
        # class_1_data = best_20_features[indices_2]

        
        # tab20c = plt.cm.get_cmap('tab20c', 20)
        # box_colors = [tab20c(2), tab20c(10),tab20c(12),tab20c(15),tab20c(7)   ]

        # # Define colors for the outlier dots
        # outlier_color = 'black'

        # # Plot box plots for each class separately
        # fig, axs = plt.subplots(num_features, 2, figsize=(12, 2*num_features))

        # for i in range(num_features):
        #     # Box plot for class 0
        #     axs[i, 0].boxplot(class_0_data[:, i], vert=False, patch_artist=True, boxprops=dict(facecolor=box_colors[i]), 
        #                     flierprops=dict(marker='o', markerfacecolor=outlier_color, markersize=5))
        #     axs[i, 0].set_title(f'Feature {i+1} (Class 0)')
        #     axs[i, 0].set_xlabel('Value')
        #     axs[i, 0].set_ylabel('')

        #     # Box plot for class 1
        #     axs[i, 1].boxplot(class_1_data[:, i], vert=False, patch_artist=True, boxprops=dict(facecolor=box_colors[i]), 
        #                     flierprops=dict(marker='o', markerfacecolor=outlier_color, markersize=5))
        #     axs[i, 1].set_title(f'Feature {i+1} (Class 1)')
        #     axs[i, 1].set_xlabel('Value')
        #     axs[i, 1].set_ylabel('')

        # plt.tight_layout()

        # plt.savefig('./box_plot_after.pdf')
        # # box plot before pca and anova -------------------
        # num_features = 5
        # selector = SelectKBest(f_classif, k=num_features)
        # selector.fit(box_plot_before[train_ind], y[train_ind])
        # best_20_features = selector.transform(box_plot_before[train_ind])
                
        
        # # Split the data based on class labels
        # c = y[train_ind]
        # print(best_20_features)
        # print(c)
        
        # indices_1 = [i for i, val in enumerate(c) if val == 1]
        # indices_2 = [i for i, val in enumerate(c) if val == 2]

        # class_0_data = best_20_features[indices_1]
        # class_1_data = best_20_features[indices_2]

        
        # # Define colors for the outlier dots
        # outlier_color = 'black'

        # # Plot box plots for each class separately
        # fig, axs = plt.subplots(num_features, 2, figsize=(12, 2*num_features))

        # for i in range(num_features):
        #     # Box plot for class 0
        #     axs[i, 0].boxplot(class_0_data[:, i], vert=False, patch_artist=True, boxprops=dict(facecolor=box_colors[i]), 
        #                     flierprops=dict(marker='o', markerfacecolor=outlier_color, markersize=5))
        #     axs[i, 0].set_title(f'Feature {i+1} (Class 0)')
        #     axs[i, 0].set_xlabel('Value')
        #     axs[i, 0].set_ylabel('')

        #     # Box plot for class 1
        #     axs[i, 1].boxplot(class_1_data[:, i], vert=False, patch_artist=True, boxprops=dict(facecolor=box_colors[i]), 
        #                     flierprops=dict(marker='o', markerfacecolor=outlier_color, markersize=5))
        #     axs[i, 1].set_title(f'Feature {i+1} (Class 1)')
        #     axs[i, 1].set_xlabel('Value')
        #     axs[i, 1].set_ylabel('')

        # plt.tight_layout()
        # plt.savefig('./box_plot_before.pdf')
        
        
        # ------------------------------------
        if model == 'MIDA':
            domain_ft = MIDA.site_information_mat(phenotype_raw, num_subjects, num_domains)
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft,
                                     domain_ft=domain_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            x_data = MIDA.MIDA(features, domain_ft, mu=best_model['mu'], h=best_model['h'], y=None)
            best_model.pop('mu')
            best_model.pop('h')

        else:
            best_model = grid_search(params, train_ind, test_ind, features, y, y_data, phenotype_ft=phenotype_ft)
            print('best parameters from 5CV grid search: \n', best_model)
            if pheno_only:
                features = phenotype_ft
            x_data = features

        if add_phenotypes:
            x_data = np.concatenate([x_data, phenotype_ft], axis=1)

        # Remove accuracy key from best model dictionary
        best_model.pop('acc')

        # Set classifier
        if algorithm == 'LR':
            clf = LogisticRegression(random_state=seed, solver='lbfgs', **best_model)

        elif algorithm == 'SVM':
            clf = svm.SVC(random_state=seed, kernel='linear', **best_model)
        else:
            clf = RidgeClassifier(random_state=seed, **best_model)

        clf.fit(x_data[train_ind, :], y[train_ind].ravel())

        # Compute the accuracy
        lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())

        # Compute the AUC
        pred = clf.decision_function(x_data[test_ind, :])
        lin_auc = roc_auc_score(y[test_ind], pred)
        
        
        # plt.show()
        
        ### ROC AUC curve for each class -------------------------------
        # y_true=y[test_ind]-1
        # y_pred=pred-1
        # fpr, tpr, _ = roc_curve(y_true, y_pred)
        # roc_auc = auc(fpr, tpr)

        # # Plot ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(f'ROC Curve on Fold {k}')
        # plt.legend(loc="lower right")

        # plt.savefig(f'./roc_curve_{k}.pdf')
        # plt.show()

        # ROC AUC for all classes in one plot ---------------------
        # y_true = y[test_ind] - 1
        # y_pred = pred - 1

        # # Calculate ROC curve for the current fold
        # fpr, tpr, _ = roc_curve(y_true, y_pred)
        # roc_auc = auc(fpr, tpr)

        # # Store fpr and tpr values for the current fold
        # all_fpr.append(fpr)
        # all_tpr.append(tpr)
        # all_auc.append(roc_auc)
        #-------------------------------

        # append accuracy and AUC to respective lists
        results_acc.append(lin_acc)
        results_auc.append(lin_auc)
        print("-" * 100)
        print("Fold number: %d" % k)
        print("Linear Accuracy: " + str(lin_acc))
        print("Linear AUC: " + str(lin_auc))
        print("-" * 100)

    # continue AUC ROC in one plot -----------
    # plt.figure()

    
    # colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # for k in range(10):
    #     plt.plot(all_fpr[k], all_tpr[k], color=colors[k], lw=2, label='Fold %d (area = %0.2f)' % (k, all_auc[k]))

    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # plt.title('ROC Curve for 10-fold Cross-Validation')
    # plt.legend(loc="lower right")
    # plt.savefig('./roc_curve_all2.pdf')
    # # plt.show()

    #------------------------
    avg_acc = np.array(results_acc).mean()
    std_acc = np.array(results_acc).std()
    avg_auc = np.array(results_auc).mean()
    std_auc = np.array(results_auc).std()
    print("accuracy average", avg_acc)
    print("standard deviation accuracy", std_acc)
    print("auc average", avg_auc)
    print("standard deviation auc", std_auc)

    # compute statistical test of independence
    if params['KHSIC'] and model == 'MIDA':
        test_stat, threshold, pval = KHSIC.hsic_gam(features, domain_ft, alph=0.05)
        pval = 1 - pval
        print('KHSIC sample value: %.2f' % test_stat, 'Threshold: %.2f' % threshold, 'p value: %.10f' % pval)

    all_results = pd.DataFrame()
    all_results['ACC'] = results_acc
    all_results['AUC'] = results_auc
    all_results.to_csv(filename + '.csv')
