---
nav_include: 4
title: Conclusions
---

## Conclusions

In this project we have developed a model able to predict Alzheimer’s disease with a 87% accuracy based on a baseline set of tests and a second evaluation 12 months later. 

In order to get there we had to get familiar with a vast amount of data and many new concepts. We started by attempting to obtain our own merged dataset because we believed on the one hand we would be more flexible, and on the other hand we would better understand the data we are working with. However, the size of the data, the variety of formats, and the level of time and understanding required made us go with the ADNI merged data set.  But at that time we had a good understanding of the data we were dealing with and we understood how strenuous this process can be for some datasets. Still a very important part of Data Science.

After doing some standard preprocessing to the data we experimented with different models, including an unsupervised classification technique never seen in class: UMAP.  We concluded that Random Forest is the best model for predicting a correct diagnosis for patients.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic</td>
      <td>0.85473</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.726351</td>
    </tr>
    <tr>
      <td>LDA</td>
      <td>0.766892</td>
    </tr>
    <tr>
      <td>QDA</td>
      <td>0.405405</td>
    </tr>
    <tr>
      <td>Decision Trees MaxDepth10</td>
      <td>0.851351</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>0.885135</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.868243</td>
    </tr>
    <tr>
      <td>SVM-GridOptimized</td>
      <td>0.868243</td>
    </tr>
    <tr>
      <td>Neural Network</td>
      <td>0.459459</td>
    </tr>
    <tr>
      <td>UMAP</td>
      <td>UMAP_svc_acc</td>
    </tr>
  </tbody>
</table>
