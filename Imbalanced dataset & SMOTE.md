__How to deal with imbalanced dataset?__ 

- Oversampling: No informaiton loss, perform better than understanding, but overfitting issues
    - overfitting: when model is too complex, training error is too small, but test error is large (the accrucay is going down)
    - underfitting: when model is too simple, both training and test errors are are large
- Undersampling: Help improve run time and storage problems but information loss, biased dataset

__About SMOTE__
- Short for synthetic minority oversampling technique
- One of the popular techniques for oversampling to solve the imbalance problem
- Aim to balance class distribution by randomly increasing minority class examples by replicating them
- In jupyter notebook, try

pip install imblearn

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

sm = SMOTE()

trainx_res, trainy_res = sm.fit_sample(trainx.astype("float"),trainy.ravel())

- if fit_sample doesn't work, try fit_resample
 
print("Before oversampling, counts of label '1': {}".format(sum(trainy==1)))

print("Before oversampling, counts of label '0': {} \n".format(sum(trainy ==0)))

print('After Oversampling, the shape of trainx: {}'.format(trainx_res.shape))

print('After Oversampling, the shape of trainy: {} \n'.format(trainy_res.shape))

print('After Oversampling, counts of label "1": {}'.format(sum(trainy_res==1)))

print('After Oversampling, counts of label "0": {}'.format(sum(trainy_res==0)))



source: <https://www.youtube.com/watch?v=dkXB8HH_4-k&ab_channel=DataMites>





