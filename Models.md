---
nav_include: 3
title: Models
---

## Models

In this section we are going to apply different models to our data in search of the bes classifier. We are facing a 3 class classification problem with the diagnosis of Alzhemer being the response variable and which can be Normal (NL), Mild Cognitive Impairment (MCI) or Dementia/Alzehimer's Disease (AD). But first we need a systematic approach to apply the different models so we are going to group some functions together:



```
### Functions for pre-processing ###

def fit_model(model, model_name: str, ADNI_df: pd.DataFrame, to_OHE: list, non_pred: list, sparse: list, viscode_select='m12',truth='DX', graphviz_plot=False):
  
  X_train, X_test, y_train, y_test = prepare_data(ADNI_df, to_OHE, non_pred, sparse)
  
  #fit model
  model = model.fit(X_train,y_train)
  
  #generate predicted values
  y_pred = model.predict(X_test)
  
  #generate plots to assess performance metrics
  make_model_plots(y_test,y_pred,model_name)
  
  if graphviz_plot:
    dot_data=export_graphviz(model, out_file=None) 


    dot_data_sized=edit_dot_string(dot_data,(15,18))

    graph = graphviz.Source(dot_data_sized)
    display(graph)
  
  return model, accuracy_score(y_test,y_pred)

def make_model_plots(y_test,y_pred,model_name: str):
  
  print('The accuracy of %s is %g\n' % (model_name, accuracy_score(y_test,y_pred)))
  
  #calculate confusion matrix and make plot
  confusion = confusion_matrix(y_test,y_pred)
  plt.figure(0)
  ax=sns.heatmap(confusion, annot=True, xticklabels=['CN', 'Dementia', 'MCI'], yticklabels=['CN', 'Dementia', 'MCI']);
  ax.set_title('%s Classifier: Confusion Matrix' % model_name);
  ax.set_xlabel('Predicted class');
  ax.set_ylabel('True class');
  
  #make plot of real vs predicted class counts
  plot_df=pd.DataFrame([y_test],index=['Class']).T
  plot_df['Value Type']='True'


  plot_df2=pd.DataFrame([y_pred],index=['Class']).T
  plot_df2['Value Type']='Predicted'

  plot_df_final=plot_df.append(plot_df2)
  plot_df_final.tail()
  
  plt.figure(1)
  ax2=sns.countplot(x='Class',data=plot_df_final,hue='Value Type');
  ax2.set_xticklabels(['CN', 'Dementia', 'MCI']);
  ax2.set_title('%s: True vs Predicted Values' % model_name);
  
  
def edit_dot_string(dot_data, size: tuple):
  
  import io
  
  newline=""
  
  s = io.StringIO(dot_data)
  for i, line in enumerate(s):
    
    if i==1:
      newline=newline + """\n ratio="fill";\nsize="%g,%g!";\n""" % size 
    
    else:
      newline=newline+line
  
  return newline
```


### Logistic Regression (Base Model)

We are going to start applying one of the most simple classification models and we are going to use it as our base model: Logistic Regresson. From there we will try different models, including one that has not been seen during class.



```
#Fit a Logistic Regression
logregcv = LogisticRegressionCV(Cs=10, cv=5, penalty='l2',solver='lbfgs', max_iter=1000, n_jobs=-1)

logregcv, logregcv_acc = fit_model(logregcv, 'Logistic Regression CV', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of Logistic Regression CV is 0.85473
    



![png](Models_files/Models_3_1.png)



![png](Models_files/Models_3_2.png)


### KNN




```
#Fit KNN Regression
knn = KNeighborsClassifier(n_neighbors=5)  

knn, knn_acc = fit_model(knn, 'KNN', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of KNN is 0.726351
    



![png](Models_files/Models_5_1.png)



![png](Models_files/Models_5_2.png)


### LDA



```
#Fit LDA Classifier
LDA = LinearDiscriminantAnalysis()

LDA, LDA_acc = fit_model(LDA, 'LDA', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of LDA is 0.766892
    



![png](Models_files/Models_7_1.png)



![png](Models_files/Models_7_2.png)


### QDA



```
#Fit QDA Classifier
QDA = QuadraticDiscriminantAnalysis()

QDA, QDA_acc = fit_model(QDA, 'QDA', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of QDA is 0.405405
    



![png](Models_files/Models_9_1.png)



![png](Models_files/Models_9_2.png)


### Decision Trees

Decision Trees are comprised of a nested series of comparisons between predictors that are compared to threshold values.  Each decision, or split, involves a single feature and a threshold value that separates the classes.  The "tree" takes an upside down appearance with the root class at the top, and subsequent branches and nodes becoming more "pure" for a specific class as we move down.  

Here we will fit a simple decision tree model to our data and stop it at a max depth of 10 to help mitigate overfitting.  We'll then graph what the tree looks like and plot the summary of the performance.



```
#lets try again with a less overfit tree
clf10 = DecisionTreeClassifier(max_depth=10)

clf10, clf10_acc = fit_model(clf10, 'Decision Tree: MaxDepth=10', ADNI_df, to_OHE, non_pred, sparse, graphviz_plot=True)

```


    The accuracy of Decision Tree: MaxDepth=10 is 0.851351
    



![svg](Models_files/Models_11_1.svg)



![png](Models_files/Models_11_2.png)



![png](Models_files/Models_11_3.png)


### Random Forest

The decision trees performed fairly well at classification, but let's see if we can do better using a random forest.  This combines random combinations of decision trees along with random subsets of predictors to create an ensemble model with increased predictive value over any of the base simple models.  



```
from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier(n_estimators=100, max_depth=None)
rf_model, rf_model_acc = fit_model(rf_model, 'Random Forest', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of Random Forest is 0.885135
    



![png](Models_files/Models_13_1.png)



![png](Models_files/Models_13_2.png)


### Neural Network



In addition to traditional regression types, we also investigated two simple neural networks.  Multi-Layer-Perceptrons known as (MLPs) use forward propagation and automatic differentiation in order to learn weights between connected node models.  This approach can be useful when working with complication correlations.

#### Preprocessing

Some slight preprocessing is required to build out a connected neural network on our data.  One of the main design choices we made was to one hot encode our three classes of DX's, use Min-Max Scalar to put all of our data in the range of [0,1] and fillna with a mean imputation.



```
viscode_select='m12'
truth='DX'

#Take only one year visits and DXs with non-NaN values
ADNI_selected = ADNI_df.loc[ADNI_df['VISCODE'] == viscode_select]
ADNI_selected = ADNI_selected.dropna(subset = [truth])

#Define predictors and output
y = ADNI_selected[truth]
X = ADNI_selected.drop(sparse,axis = 1)
X = X.drop(non_pred,axis = 1)

#Replace NaNs in non-categorical data
X_float=X.drop(to_OHE,axis=1)
X_float = X_float.fillna(X_float.mean())

scaled_values = MinMaxScaler().fit_transform(X_float)
X_float = pd.DataFrame(scaled_values, index=X_float.index, columns=X_float.columns)

#One hot encode where necessary
X_cat = pd.get_dummies(X[to_OHE])
y = pd.get_dummies(y)

X = pd.concat([X_float,X_cat], axis=1)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```


Once our data has been preprocessed appropriately, we first used a very simple fully connected NN using keras.

#### MLP

We selected an appropriate input shape and several hidden layers of various dense sizes, and an output into 3 classes to predict the different Dxs:



```
inp = Input(shape = (X_train.shape[1],))
x = Dense(200, activation='relu')(inp)
x = Dense(100, activation='relu')(x)
x = Dense(50, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(3, activation='sigmoid')(x)

model = Model(inputs=inp, outputs= x)

model.summary()
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 53)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               10800     
    _________________________________________________________________
    dense_2 (Dense)              (None, 100)               20100     
    _________________________________________________________________
    dense_3 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dense_4 (Dense)              (None, 128)               6528      
    _________________________________________________________________
    dense_5 (Dense)              (None, 3)                 387       
    =================================================================
    Total params: 42,865
    Trainable params: 42,865
    Non-trainable params: 0
    _________________________________________________________________




```
model.compile(optimizer='Adam', loss='mean_absolute_error', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs = 20, validation_split=0.2)

NN_acc = model.evaluate(X_test, y_test)[1]
```


    Train on 945 samples, validate on 237 samples
    Epoch 1/20
    945/945 [==============================] - 1s 644us/step - loss: 0.3701 - acc: 0.4646 - val_loss: 0.3334 - val_acc: 0.4430
    Epoch 2/20
    945/945 [==============================] - 0s 53us/step - loss: 0.3334 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 3/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 4/20
    945/945 [==============================] - 0s 49us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 5/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 6/20
    945/945 [==============================] - 0s 55us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 7/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 8/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 9/20
    945/945 [==============================] - 0s 51us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 10/20
    945/945 [==============================] - 0s 51us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 11/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 12/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 13/20
    945/945 [==============================] - 0s 53us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 14/20
    945/945 [==============================] - 0s 51us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 15/20
    945/945 [==============================] - 0s 55us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 16/20
    945/945 [==============================] - 0s 51us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 17/20
    945/945 [==============================] - 0s 53us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 18/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 19/20
    945/945 [==============================] - 0s 51us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    Epoch 20/20
    945/945 [==============================] - 0s 54us/step - loss: 0.3333 - acc: 0.4646 - val_loss: 0.3333 - val_acc: 0.4430
    296/296 [==============================] - 0s 35us/step




```
NN_acc
```





    0.4594594594594595



Our Neural Network once trained created a decently accurate model to predict our patient DX.  This could be a useful tool, however due to the lack of interpretability of NNs, very little information can be passed back to the physician regarding the importance of the different predictors.

### SVM

Support Vector Machines seek to separate date by constructing optimal hyperplanes.  Hyperplanes are n-dimensional separators that are placed to maximize the distance between the decision boundary (the hyperplane) and the classes.  The data points closest to the decision boundary are called support vectors and they are what determine the position and orientation of the hyperplane.  The distance of the support vectors to the decision boundary is minimized using a cost function.


Let's fit a simple Support Vector Classifier and see how it performs on our data. 



```
svc = LinearSVC(C=100, dual=False, class_weight='balanced', max_iter=10000, random_state=42)
svc, svc_acc = fit_model(svc, 'SVM', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of SVM is 0.868243
    



![png](Models_files/Models_27_1.png)



![png](Models_files/Models_27_2.png)


Perhaps we can do better by tuning our C penalty parameter using GridSearchCV.



```
# Classification with a linear SVM, optimizing C using GridSearchCV
svc_grid = LinearSVC(dual=False, random_state=42, class_weight='balanced')
params_grid = {"C": [10**k for k in range(-3, 4)]}
svc_grid = GridSearchCV(svc_grid, params_grid)

svc_grid, svc_grid_acc = fit_model(svc_grid, 'SVM - GridSearch Optimized', ADNI_df, to_OHE, non_pred, sparse)
```


    The accuracy of SVM - GridSearch Optimized is 0.868243
    



![png](Models_files/Models_29_1.png)



![png](Models_files/Models_29_2.png)


### Dimensionality Reduction Exploration: UMAP

Since we are dealing with a large number of features, it may be helpful to explore dimensionality reduction techniques to aid in both visualization and improving the performance of one of our models.  The UMAP method produces similar results to the PCA method we learned in class, but goes about reducing dimensions in a very different way.  Most dimension reduction techniques fall into two camps: matrix factorization (like PCA) and neighbor graphs (like UMAP).  UMAP is a manifold method that takes into account local relationships between neighboring points while preserving the overall global structure of the data.  More information can be found in the cited paper in the **Literature** section.

First, let's use UMAP to reduce all of our dimensions down to 2 for easy plotting and take a look at our data.  2-dimensional data is easily plotted and interpreted by people, so it's a sensible starting point



```
def plot_umap(ADNI_df: pd.DataFrame, to_OHE, non_pred, sparse):
  
  #obtain our training and test data as usual
  X_train, X_test, y_train, y_test = prepare_data(ADNI_df, to_OHE, non_pred, sparse)
  
  #stack all available data together for embedding and visualization
  data=np.vstack([X_train,X_test])
  target=np.hstack([y_train,y_test])
  embedding=UMAP(random_state=42, n_components=2,n_neighbors=500,min_dist=1,metric='braycurtis').fit_transform(data)
  
  fig, ax = plt.subplots(1, figsize=(14, 10))
  ax.scatter(*embedding.T, c=target, cmap='plasma',alpha=0.7)
  
  #insert patches for legend
  blue_patch = mpatches.Patch(color='#002f7c', label='Dementia')
  red_patch = mpatches.Patch(color='#c6395a', label='MCI')
  yellow_patch = mpatches.Patch(color='#ffff0c', label='Cognitively Normal')
  plt.legend(handles=[blue_patch,red_patch,yellow_patch])
  
plot_umap(ADNI_df, to_OHE, non_pred, sparse)
```



![png](Models_files/Models_33_0.png)


Next, we will try to use UMAP and GridSearchCv to improve on the Support Vector Classifier model we fit previously.



```
def svc_Umap(ADNI_df: pd.DataFrame, to_OHE, non_pred, sparse):
  X_train, X_test, y_train, y_test = prepare_data(ADNI_df, to_OHE, non_pred, sparse)

  # Classification with a linear SVM, as done before
  svc_grid_umap = LinearSVC(dual=False, random_state=42, class_weight='balanced')
  params_grid = {"C": [10**k for k in range(-2, 3)]}
  clf = GridSearchCV(svc_grid_umap, params_grid, cv=5, n_jobs=-1)
  clf.fit(X_train, y_train)

  # Transform with UMAP followed by SVM classification using pipeline
  umap = UMAP(random_state=4242, metric='minkowski', min_dist=0.5)
  pipeline = Pipeline([("umap", umap),
                     ("svc", svc_grid_umap)])
  params_grid_pipeline = {"umap__n_neighbors": [20,500],
                        "umap__n_components": [5, 10],
                        "svc__C": [10**k for k in range(-2, 3)]}

  clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline, cv=5, n_jobs=-1)
  clf_pipeline.fit(X_train, y_train)

  print("\nSVM accuracy on the test set with raw data: {:.3f}".format(
    clf.score(X_test, y_test)))

  print("SVM accuracy on the test set with UMAP transformation: {:.3f}".format(
    clf_pipeline.score(X_test, y_test)))
  
  return clf_pipeline.score(X_test, y_test)




UMAP_svc_acc = svc_Umap(ADNI_df, to_OHE, non_pred, sparse)
```


    
    SVM accuracy on the test set with raw data: 0.868
    SVM accuracy on the test set with UMAP transformation: 0.713


It seems that UMAP made things a bit worse for prediction in this specific case.  UMAP is usually used when there are thousands to tens of thousand predictors.  When we intially started on the project, it appeared that we would at least have thousands of predictors to model with.  But due to knowlege of the data, practical concerns and limitations of the dataset structure, we ended up with fewer predictors than we initially anticipated, making UMAP unecessary for this specific application.  It is, however, a very useful technique for true high-dimensionality problems that arise in DNA and RNA sequencing applications for which this method was initially developed for.
