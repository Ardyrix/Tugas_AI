# Hermon Jay 14-10-2017
# klasifikasi jenis kelamin dengan 
# Decision Tree, SVM, KNN, dan Naive Bayes

from sklearn import tree
from sklearn.metrics import accuracy_score

# model untuk ketiga classifier
cDT = tree.DecisionTreeClassifier()

# data latih
# [tingi, berat, ukuran_sepatu]
X = [[30,64,1], [30,62,3], [30,65,0], [31,59,2], [31,65,4],
     [33,58,10], [33,60,0], [34,59,0], [34,66,9], [34,58,30],
     [34,60,1]]

Y = ['1', '1', '1', '1', '1', '1', '1', '1',
     '2', '2', '1']

# latih classifier
cDT = cDT.fit(X, Y)

# data test
X_test = [[34,69,10], [35,33,22], [34,67,7], [34,60,0], [35,64,13],
         [35,63,0], [36,60,1], [36,69,0], [37,60,0], [37,63,0],
         [37,59,1]]
Y_test = ['1', '2', '1', '2', '1', '1', '1', '1',
          '1', '1', '1']

# prediksi data test
Y_DT = cDT.predict(X_test)


# print prediksi
#print("Prediksi Decision Tree : ", Y_DT)
#print("Prediksi SVM : ", Y_SVM)
#print("Prediksi KNN : ", Y_KNN)
#print("Prediksi Naive Bayes : ", Y_NB)

# print akurasi
print("Akurasi Decision Tree : ", accuracy_score(Y_test, Y_DT))