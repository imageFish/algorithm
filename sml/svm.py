from sklearn import datasets, svm
from datetime import datetime
iris = datasets.load_iris()
digits = datasets.load_digits()
for c in range(-5, 16):
    c = pow(2, c)
    for gamma in range(-15, 4):
        st = datetime.now()
        gamma = pow(2, gamma)
        clf = svm.SVC(gamma=0.001, C=100)
        clf.fit(digits.data[:-1], digits.target[:-1])
        pred = clf.predict(digits.data[-1:])
        print(pred, pred[0]==digits.target[-1], str(datetime.now()-st))
print('done')