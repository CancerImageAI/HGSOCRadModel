# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:20:33 2021

@author: DELL
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np
import rpy2.robjects as robj
r = robj.r
from rpy2.robjects.packages import importr

def roc_test_r(targets_1, scores_1, targets_2, scores_2, method='delong'):
    # method: “delong”, “bootstrap” or “venkatraman”
    importr('pROC')
    robj.globalenv['targets_1'] = targets_1 = robj.FloatVector(targets_1)
    robj.globalenv['scores_1'] = scores_1 = robj.FloatVector(scores_1)
    robj.globalenv['targets_2'] = targets_2 = robj.FloatVector(targets_2)
    robj.globalenv['scores_2'] = scores_2 = robj.FloatVector(scores_2)

    r('roc_1 <- roc(targets_1, scores_1)')
    r('roc_2 <- roc(targets_2, scores_2)')
    r('result = roc.test(roc_1, roc_2, method="%s")' % method)
    p_value = r('p_value = result$p.value')
    return np.array(p_value)[0]


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

if __name__ == '__main__': 
    ## T1 FUSCC dataset
    FUSCC_T1file = open('./Result/Fudan Cancer Hospital/T1_Feature.csv')
    FUSCC_T1list = pd.read_csv(FUSCC_T1file)
    FUSCC_T1list = FUSCC_T1list.fillna('0')
    FUSCC_T1class = FUSCC_T1list['resistance']
    FUSCC_T1patientName = np.array(FUSCC_T1list['Name'])
    FUSCC_T1features = FUSCC_T1list.values[:,:-11]
    FUSCC_T1featureName = list(FUSCC_T1list.head(0))[:-11]
    FUSCC_T1class = [int(i) if int(i)==1 else 0 for i in FUSCC_T1class]
    FUSCC_Diameter = FUSCC_T1list['original_shape_Maximum3DDiameter']
    FUSCC_Diameter = [1 if int(i)>=50 else 0 for i in FUSCC_Diameter]

    #T1 G&O dataset
    GOH_T1file = open('./Result/G&O Hospital/T1_Feature.csv')
    GOH_T1list = pd.read_csv(GOH_T1file)
    GOH_T1list = GOH_T1list.fillna('0')
    GOH_T1class = GOH_T1list['resistance']
    GOH_T1patientName = np.array(GOH_T1list['Name'])
    GOH_T1features = GOH_T1list.values[:,:-11]
    GOH_T1featureName = list(GOH_T1list.head(0))[:-11]
    GOH_T1class = [int(i) if int(i)==1 else 0 for i in GOH_T1class]
    GOH_Diameter = GOH_T1list['original_shape_Maximum3DDiameter']
    GOH_Diameter = [1 if int(i)>=50 else 0 for i in GOH_Diameter]

    ## Nantong Center Hospital
    NTCH_T1file = open('./Result/Nantong Cancer Hospital/T1_Feature.csv')
    NTCH_T1list = pd.read_csv(NTCH_T1file)
    NTCH_T1list = NTCH_T1list.fillna('0')
    NTCH_T1class = NTCH_T1list['resistance']
    NTCH_T1patientName = np.array(NTCH_T1list['Name'])
    NTCH_T1features = NTCH_T1list.values[:,:-11]
    NTCH_T1featureName = list(NTCH_T1list.head(0))[:-11]
    NTCH_T1class = [int(i) if int(i)==1 else 0 for i in NTCH_T1class]
    NTCH_Diameter = NTCH_T1list['original_shape_Maximum3DDiameter']
    NTCH_Diameter = [1 if int(i)>=50 else 0 for i in NTCH_Diameter]

    ## Zhongshan Hospital
    ZSH_T1file = open('./Result/Zhongshan Hospital/T1_Feature.csv')
    ZSH_T1list = pd.read_csv(ZSH_T1file)
    ZSH_T1list = ZSH_T1list.fillna('0')
    ZSH_T1class = ZSH_T1list['resistance']
    ZSH_T1patientName = np.array(ZSH_T1list['Name'])
    ZSH_T1features = ZSH_T1list.values[:,:-11]
    ZSH_T1featureName = list(ZSH_T1list.head(0))[:-11]
    ZSH_T1class = [int(i) if int(i)==1 else 0 for i in ZSH_T1class]
    ZSH_Diameter = ZSH_T1list['original_shape_Maximum3DDiameter']
    ZSH_Diameter = [1 if int(i)>=50 else 0 for i in ZSH_Diameter]

    ## Testing Dataset
    Testing_T1features = np.vstack((GOH_T1features,NTCH_T1features,ZSH_T1features))
    Testing_T1Class = GOH_T1class+NTCH_T1class+ZSH_T1class
    Testing_Diameter = GOH_Diameter+NTCH_Diameter+ZSH_Diameter
    
    scaler = StandardScaler()
    FUSCC_T1features = scaler.fit_transform(FUSCC_T1features)
    Testing_T1features = scaler.transform(Testing_T1features)
    clf =  LinearSVC(penalty="l1", dual=False, random_state=0).fit(FUSCC_T1features, FUSCC_T1class)
    model_T1 = SelectFromModel(clf, prefit=True,max_features = 11)
    train_T1 = model_T1.transform(FUSCC_T1features)
    test_T1 = model_T1.transform(Testing_T1features)
    x_train_T1, y_train_T1 = SMOTE(sampling_strategy='auto',k_neighbors=5,random_state=0).fit_resample(train_T1, FUSCC_T1class)
    clf_T1 = svm.SVC(kernel="poly",  probability=True, random_state=0)
    clf_T1.fit(x_train_T1, y_train_T1)
    
    
    prob_T1 = clf_T1.predict_proba(test_T1)[:,1]
    pred_lable_T1 = clf_T1.predict(test_T1)
    fpr_T1,tpr_T1,threshold_T1 = roc_curve(Testing_T1Class, np.array(prob_T1)) ###计算真正率和假正率
    auc_T1 = auc(fpr_T1,tpr_T1)
    l_T1, h_T1, std_T1 = confindence_interval_compute(np.array(prob_T1), np.array(Testing_T1Class))
    print('T1 Feature AUC:%.2f'%auc_T1,'+/-%.2f'%std_T1,'  95% CI:[','%.2f,'%l_T1,'%.2f'%h_T1,']')
    print('T1 Feature ACC:%.4f'%accuracy_score(Testing_T1Class,pred_lable_T1)) 
    
    FUSCC_test_T1 = train_T1
    FUSCC_prob_T1 = clf_T1.predict_proba(FUSCC_test_T1)[:,1]
    FUSCC_pred_lable_T1 = clf_T1.predict(FUSCC_test_T1)
    FUSCC_fpr_T1,FUSCC_tpr_T1,FUSCC_threshold_T1 = roc_curve(FUSCC_T1class, np.array(FUSCC_prob_T1)) ###计算真正率和假正率
    FUSCC_auc_T1 = auc(FUSCC_fpr_T1,FUSCC_tpr_T1)
    FUSCC_l_T1, FUSCC_h_T1, FUSCC_std_T1 = confindence_interval_compute(np.array(FUSCC_prob_T1), np.array(FUSCC_T1class))
    print('FUSCC T1 Feature AUC:%.2f'%FUSCC_auc_T1,'+/-%.2f'%FUSCC_std_T1,'  95% CI:[','%.2f,'%FUSCC_l_T1,'%.2f'%FUSCC_h_T1,']')
    print('FUSCC T1 Feature ACC:%.4f'%accuracy_score(FUSCC_T1class,FUSCC_pred_lable_T1))
    
    lw = 1.5
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 12,}
    plt.rc('font', **font)
    idx = np.arange(0, FUSCC_T1features.shape[1])  #create an index array
    indices = idx[model_T1.get_support() == True]  #get index positions of kept features
    print(np.array(FUSCC_T1featureName)[indices])
    feature_names = [r'T1-F'+str(i) for i in range(1,12)]
    selected_feature = FUSCC_T1features[:,indices]
    Class_Type = [i if i==0  else 'Resistance' for i in FUSCC_T1class]
    Class_Type = [i if i=='Resistance' else 'Non-resistance' for i in Class_Type]
    SelectedFeatures = {}
    SelectedFeatures['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    SelectedFeatures['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    SelectedFeatures['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    SelectedFeatures = pd.DataFrame.from_dict(SelectedFeatures)
    plt.figure(figsize=(7,4))
    sns.boxplot(x="FeatureName", y="Feature",hue='Class_Type',data=SelectedFeatures,
                     palette='Set1')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k')
    plt.subplots_adjust(top=0.985,bottom=0.06,left=0.05,right=0.99,hspace=0,wspace=0)
    
    
    ## T2 FUSCC dataset
    FUSCC_T2file = open('./Result/Fudan Cancer Hospital/T2_Feature.csv')
    FUSCC_T2list = pd.read_csv(FUSCC_T2file)
    FUSCC_T2list = FUSCC_T2list.fillna('0')
    FUSCC_T2class = FUSCC_T2list['resistance']
    FUSCC_T2patientName = np.array(FUSCC_T2list['Name'])
    FUSCC_T2features = FUSCC_T2list.values[:,:-11]
    FUSCC_T2featureName = list(FUSCC_T2list.head(0))[:-11]
    FUSCC_T2class = [int(i) if int(i)==1 else 0 for i in FUSCC_T2class]
    #T2 G&O dataset
    GOH_T2file = open('./Result/G&O Hospital/T2_Feature.csv')
    GOH_T2list = pd.read_csv(GOH_T2file)
    GOH_T2list = GOH_T2list.fillna('0')
    GOH_T2class = GOH_T2list['resistance']
    GOH_T2patientName = np.array(GOH_T2list['Name'])
    GOH_T2features = GOH_T2list.values[:,:-11]
    GOH_T2featureName = list(GOH_T2list.head(0))[:-11]
    GOH_T2class = [int(i) if int(i)==1 else 0 for i in GOH_T2class]
    ## Nantong Center Hospital
    NTCH_T2file = open('./Result/Nantong Cancer Hospital/T2_Feature.csv')
    NTCH_T2list = pd.read_csv(NTCH_T2file)
    NTCH_T2list = NTCH_T2list.fillna('0')
    NTCH_T2class = NTCH_T2list['resistance']
    NTCH_T2patientName = np.array(NTCH_T2list['Name'])
    NTCH_T2features = NTCH_T2list.values[:,:-11]
    NTCH_T2featureName = list(NTCH_T2list.head(0))[:-11]
    NTCH_T2class = [int(i) if int(i)==1 else 0 for i in NTCH_T2class]
    ## Zhongshan Hospital
    ZSH_T2file = open('./Result/Zhongshan Hospital/T2_Feature.csv')
    ZSH_T2list = pd.read_csv(ZSH_T2file)
    ZSH_T2list = ZSH_T2list.fillna('0')
    ZSH_T2class = ZSH_T2list['resistance']
    ZSH_T2patientName = np.array(ZSH_T2list['Name'])
    ZSH_T2features = ZSH_T2list.values[:,:-11]
    ZSH_T2featureName = list(ZSH_T2list.head(0))[:-11]
    ZSH_T2class = [int(i) if int(i)==1 else 0 for i in ZSH_T2class]
    
    ## Testing Dataset
    Testing_T2features = np.vstack((GOH_T2features,NTCH_T2features,ZSH_T2features))
    Testing_T2Class = GOH_T2class+NTCH_T2class+ZSH_T2class

    scaler = StandardScaler()
    FUSCC_T2features = scaler.fit_transform(FUSCC_T2features)
    Testing_T2features = scaler.transform(Testing_T2features)
    clf =  LinearSVC(penalty="l1", dual=False, random_state=0).fit(FUSCC_T2features, FUSCC_T2class)
    model_T2 = SelectFromModel(clf, prefit=True,max_features = 6, threshold=1e-9) 
    train_T2 = model_T2.transform(FUSCC_T2features)
    test_T2 = model_T2.transform(Testing_T2features)
    x_train_T2, y_train_T2 = SMOTE(sampling_strategy='auto',k_neighbors=5,random_state=0).fit_resample(train_T2, FUSCC_T2class)
    clf_T2 = svm.SVC(kernel="poly",  probability=True, random_state=0)
    clf_T2.fit(x_train_T2, y_train_T2)
    
    prob_T2 = clf_T2.predict_proba(test_T2)[:,1]
    pred_lable_T2 = clf_T2.predict(test_T2)
    fpr_T2,tpr_T2,threshold_T2 = roc_curve(Testing_T2Class, np.array(prob_T2)) ###计算真正率和假正率
    auc_T2 = auc(fpr_T2,tpr_T2)
    l_T2, h_T2, std_T2 = confindence_interval_compute(np.array(prob_T2), np.array(Testing_T2Class))
    print('T2 Feature AUC:%.2f'%auc_T2,'+/-%.2f'%std_T2,'  95% CI:[','%.2f,'%l_T2,'%.2f'%h_T2,']')
    print('T2 Feature ACC:%.4f'%accuracy_score(Testing_T2Class,pred_lable_T2)) 
    
    idx = np.arange(0, FUSCC_T2features.shape[1])  #create an index array
    indices = idx[model_T2.get_support() == True]  #get index positions of kept features
    print(np.array(FUSCC_T1featureName)[indices])
    feature_names = [r'T2-F'+str(i) for i in range(1,7)]
    selected_feature = FUSCC_T2features[:,indices]
    Class_Type = [i if i==0  else 'Resistance' for i in FUSCC_T2class]
    Class_Type = [i if i=='Resistance' else 'Non-resistance' for i in Class_Type]
    SelectedFeatures = {}
    SelectedFeatures['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    SelectedFeatures['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    SelectedFeatures['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    SelectedFeatures = pd.DataFrame.from_dict(SelectedFeatures)
    plt.figure(figsize=(7,4))
    sns.boxplot(x="FeatureName", y="Feature",hue='Class_Type',data=SelectedFeatures,
                     palette='Set1')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k')
    plt.subplots_adjust(top=0.985,bottom=0.06,left=0.05,right=0.99,hspace=0,wspace=0)    
        
    FUSCC_test_T2 = train_T2
    FUSCC_prob_T2 = clf_T2.predict_proba(FUSCC_test_T2)[:,1]
    FUSCC_pred_lable_T2 = clf_T2.predict(FUSCC_test_T2)
    FUSCC_fpr_T2,FUSCC_tpr_T2,FUSCC_threshold_T2 = roc_curve(FUSCC_T2class, np.array(FUSCC_prob_T2)) ###计算真正率和假正率
    FUSCC_auc_T2 = auc(FUSCC_fpr_T2,FUSCC_tpr_T2)
    FUSCC_l_T2, FUSCC_h_T2, FUSCC_std_T2 = confindence_interval_compute(np.array(FUSCC_prob_T2), np.array(FUSCC_T2class))
    print('FUSCC T2 Feature AUC:%.2f'%FUSCC_auc_T2,'+/-%.2f'%FUSCC_std_T2,'  95% CI:[','%.2f,'%FUSCC_l_T2,'%.2f'%FUSCC_h_T2,']')
    print('FUSCC T2 Feature ACC:%.4f'%accuracy_score(FUSCC_T2class,FUSCC_pred_lable_T2)) 
    FUSCC_Fusion = np.zeros([len(FUSCC_prob_T1),2])
    FUSCC_Fusion[:,0] = np.array(FUSCC_prob_T1)
    FUSCC_Fusion[:,1] = np.array(FUSCC_prob_T2)
    FUSCC_Fusion_min = FUSCC_Fusion.min(1)
    auc_min = roc_auc_score(np.array(FUSCC_T2class),FUSCC_Fusion_min)
    auc_fl_min, auc_fh_min, auc_fstd_min = confindence_interval_compute(np.array(FUSCC_Fusion_min), np.array(FUSCC_T2class))
    print('FUSCC Fusion AUC:%.2f'%auc_min,'+/-%.2f'%auc_fstd_min,'  95% CI:[','%.2f,'%auc_fl_min,'%.2f'%auc_fh_min,']')

    ## Fusion
    scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for scale in scales:
        prob_fusion = scale*np.array(prob_T1)+(1-scale)*np.array(prob_T2)
        auc_value = roc_auc_score(np.array(Testing_T1Class),prob_fusion)
        auc_fl, auc_fh, auc_fstd = confindence_interval_compute(np.array(prob_fusion), np.array(Testing_T1Class))
        print('Fusion Scale',scale,'AUC:%.4f'%auc_value,'+/-%.2f'%auc_fstd,
              '  95% CI:[','%.2f,'%auc_fl,'%.2f'%auc_fh,']')      
    Fusion = np.zeros([len(prob_T1),2])
    Fusion[:,0] = np.array(prob_T1)
    Fusion[:,1] = np.array(prob_T2)
    Fusion_min = Fusion.min(1)
    Fusion_max = Fusion.max(1)

    auc_min = roc_auc_score(np.array(Testing_T1Class),Fusion_min)
    auc_fl_min, auc_fh_min, auc_fstd_min = confindence_interval_compute(np.array(Fusion_min), np.array(Testing_T1Class))
    print('Min Fusion AUC:%.2f'%auc_min,'+/-%.2f'%auc_fstd_min,'  95% CI:[','%.2f,'%auc_fl_min,'%.2f'%auc_fh_min,']')
    
    auc_max = roc_auc_score(np.array(Testing_T1Class),Fusion_max)
    auc_fl_max, auc_fh_max,auc_fstd_max = confindence_interval_compute(np.array(Fusion_max), np.array(Testing_T1Class))
    print('Max Fusion AUC:%.2f'%auc_max,'+/-%.2f'%auc_fstd_max,'  95% CI:[','%.2f,'%auc_fl_max,'%.2f'%auc_fh_max,']')
    
    
    FUSCC_age = FUSCC_T1list['age']
    FUSCC_stage = np.array(FUSCC_T1list['stage'])
    FUSCC_stage = [3 if i in ['III','IIIa','IIIb','IIIc'] else i for i in FUSCC_stage]
    FUSCC_stage = [4 if i in ['IVa','IVb','IV'] else i for i in FUSCC_stage]
    FUSCC_residual = FUSCC_T1list['residual']
    FUSCC_residual = [0 if i=="R0" else i for i in FUSCC_residual]
    FUSCC_residual = [1 if i=="R1" else i for i in FUSCC_residual]
    FUSCC_residual = [2 if i=="R2" else i for i in FUSCC_residual]
    FUSCC_mass_characteristic = FUSCC_T1list['mass_characteristic']
    FUSCC_bilaterality = FUSCC_T1list['bilaterality']
    FUSCC_Diameter = FUSCC_Diameter
    FUSCC_CA125 = FUSCC_T1list['CA125']
   
    FUSCC_Clinical = []
    FUSCC_Clinical.append(FUSCC_age)
    FUSCC_Clinical.append(FUSCC_stage)
    FUSCC_Clinical.append(FUSCC_residual)
    FUSCC_Clinical.append(FUSCC_mass_characteristic)
    FUSCC_Clinical.append(FUSCC_bilaterality)
    FUSCC_Clinical.append(FUSCC_Diameter)
    FUSCC_Clinical.append(FUSCC_CA125)
    FUSCC_Clinical = np.array(FUSCC_Clinical).transpose((1,0))
    
    Test_age = list(GOH_T1list['age'])+list(NTCH_T1list['age'])+list(ZSH_T1list['age'])
    Test_stage = np.array(list(GOH_T1list['stage'])+list(NTCH_T1list['stage'])+list(ZSH_T1list['stage']))
    Test_stage = [3 if i in ['III','IIIa','IIIb','IIIc','Ⅲc','Ⅲa','Ⅲ'] else i for i in Test_stage]
    Test_stage = [4 if i in ['IVa','IVb','IV','Ⅳ','Ⅳb'] else i for i in Test_stage]
    Test_residual = list(GOH_T1list['residual'])+list(NTCH_T1list['residual'])+list(ZSH_T1list['residual'])
    Test_residual = [0 if i=="R0" else i for i in Test_residual]
    Test_residual = [1 if i=="R1" else i for i in Test_residual]
    Test_residual = [2 if i=="R2" else i for i in Test_residual]
    Test_mass_characteristic = list(GOH_T1list['mass_characteristic'])+list(NTCH_T1list['mass_characteristic'])+list(ZSH_T1list['mass_characteristic'])
    Test_bilaterality = list(GOH_T1list['bilaterality'])+list(NTCH_T1list['bilaterality'])+list(ZSH_T1list['bilaterality'])
    Test_Diameter = Testing_Diameter
    Test_CA125 = list(GOH_T1list['CA125'])+list(NTCH_T1list['CA125'])+list(ZSH_T1list['CA125'])
    Test_Clinical = []
    Test_Clinical.append(Test_age)
    Test_Clinical.append(Test_stage)
    Test_Clinical.append(Test_residual)
    Test_Clinical.append(Test_mass_characteristic)
    Test_Clinical.append(Test_bilaterality)
    Test_Clinical.append(Test_Diameter)
    Test_Clinical.append(Test_CA125)
    Test_Clinical = np.array(Test_Clinical).transpose((1,0))
    clf =  LinearSVC(penalty="l1", dual=False, random_state=0).fit(FUSCC_Clinical, FUSCC_T1class)
    model_Clinical = SelectFromModel(clf, prefit=True, max_features = 3, threshold=1e-9)
    train_Clinical = model_Clinical.transform(FUSCC_Clinical)
    test_Clinical = model_Clinical.transform(Test_Clinical)
    x_train_Clinical, y_train_Clinical = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_Clinical, FUSCC_T1class)
    clf_Clinical = svm.SVC(kernel="poly",  probability=True, random_state=0)
    clf_Clinical.fit(x_train_Clinical, y_train_Clinical)
    
    prob_Clinical = clf_Clinical.predict_proba(test_Clinical)[:,1]
    pred_lable_Clinical = clf_Clinical.predict(test_Clinical)
    fpr_Clinical,tpr_Clinical,threshold_Clinical = roc_curve(Testing_T1Class, np.array(prob_Clinical)) ###计算真正率和假正率
    auc_Clinical = auc(fpr_Clinical,tpr_Clinical)
    l_Clinical, h_Clinical, std_Clinical = confindence_interval_compute(np.array(prob_Clinical), np.array(Testing_T1Class))
    print('Clinical Feature AUC:%.2f'%auc_Clinical,'+/-%.2f'%std_Clinical,'  95% CI:[','%.2f,'%l_Clinical,'%.2f'%h_Clinical,']')
    print('Clinical Feature ACC:%.4f'%accuracy_score(Testing_T1Class,pred_lable_Clinical)) 
    #Clinical VS Radiomics
    print(roc_test_r(np.array(Testing_T1Class), Fusion_min, np.array(Testing_T1Class),prob_Clinical))
    # T1 VS ALL
    print(roc_test_r(np.array(Testing_T1Class), Fusion_min,np.array(Testing_T1Class), prob_T1))
    #T2 VS ALL
    print(roc_test_r(np.array(Testing_T1Class), Fusion_min,np.array(Testing_T1Class), prob_T2))
    #T1 VS T2
    print(roc_test_r(np.array(Testing_T1Class), prob_T1,np.array(Testing_T2Class), prob_T2))
