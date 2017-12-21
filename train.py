from ensemble import AdaBoostClassifier
from feature import NPDFeature
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle


def pre_process(dir, file):
    features = np.array([])
    for filename in os.listdir(dir):
        img = Image.open(os.path.join(dir, filename))
        resize_img = img.resize((24, 24))
        gray_img = np.array(resize_img.convert("L"))
        feature = NPDFeature(gray_img).extract()
        features = np.append(features, feature).reshape(-1, 165600)
        print(features.shape)
    pass
    with open(file, "wb") as f:
        pickle.dump(features, f)


pre_process('./datasets/original/face', "face.npy")
pre_process('./datasets/original/nonface', "nonface.npy")

if __name__ == "__main__":
    # write your code here
    X1 = pickle.load(open("face.npy", "rb"))
    y1 = np.ones(500)
    X2 = pickle.load(open("nonface.npy", "rb"))
    y2 = -1 * np.ones(500)
    X = np.append(X1, X2).reshape(-1, 165600)
    y = np.append(y1, y2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), 10)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_val)
    accuracy = np.mean((y_val == y_pre))
    print("acc = ", accuracy)
    report = classification_report(y_val, y_pre, labels=[1, -1],\
                                   target_names=['face', 'nonface'],\
                                   digits=4)
    with open('./report.txt', 'w') as f:
        f.write(report)
    pass

