from keras import Sequential
from keras.layers import Dense, Bidirectional, Embedding, Dropout, LSTM
from keras.optimizers import Adam
import preprocessing
import pandas as pd
from keras.utils.np_utils import to_categorical
from word_embedding.Word2Vec import word2vec_weight
from models.Machine_Learning.TF_IDF import TF_IDF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("Data loading...")
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/development.csv')
print("Data loading is done!")

print("Sentence cut...")
train_sentence = preprocessing.data_preprocessing(train_df['joke'])
test_sentence = preprocessing.data_preprocessing(test_df['joke'])
print("Sentence cut is done!")

# # TF-IDF对句子进行向量化
# X = TF_IDF(train_sentence)
# test = TF_IDF(test_sentence)

"""Word2Vec, 返回词向量矩阵，对训练集测试集进行padding"""
X, X_test, embedding_matrix, vocab_size = word2vec_weight(train_sentence, test_sentence)

"""数据切分为训练集、验证集"""
# 传统机器学习
y = train_df['label']
X_train, X_dev, y_train, y_dev = model_selection.train_test_split(X, y, test_size=0.2)
print("划分训练集测试集完成！")

"""转换为one-hot编码"""
y_train = to_categorical(y_train)

# """Logistic Regression"""
# clf = LogisticRegression(multi_class='ovr', solver='sag', class_weight='balanced')
# clf.fit(X_train, y_train)
#
# """Decision Tree"""
# clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# clf.fit(X_train, y_train)
#
# """Random Forest"""
# clf = RandomForestClassifier(n_estimators=15)
# clf.fit(X_train, y_train)

# """GBDT"""
# clf = GradientBoostingClassifier()
# clf.fit(X_train, y_train)

# """XGBoost"""
# clf = XGBClassifier()
# clf.fit(X_train, y_train)
#
# """Lightgbm"""
# clf = LGBMClassifier()
# clf.fit(X_train, y_train)
#
# """SVM"""
# clf = SVR()
# clf.fit(X_train, y_train)
#
# """Kfold交叉验证"""
# skfold = StratifiedKFold(n_splits=5, random_state=1)
# for train_index, test_index in skfold.split(X, y):
#     clone_clf = clone(clf)
#     X_train_folds = X[train_index]
#     y_train_folds = y[train_index]
#     X_test_fold = X[test_index]
#     y_test_fold = y[test_index]
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     f1 = f1_score(y_test_fold, y_pred)
#     print("F1预测值为"+str(f1))

# """网格搜索调参，每个参数进行调整"""
# """GBDT调参作为例子"""
# param_n_estimators = {'n_estimators': range(20, 81, 10)}
# grid_search_1 = GridSearchCV(estimator=GradientBoostingClassifier(),
#                              param_grid=param_n_estimators,
#                              iid=False,
#                              scoring='roc_auc',
#                              cv=5)
# grid_search_1.fit(X_train, y_train)
# print(grid_search_1.best_params_, grid_search_1.best_score_)

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1,
                    output_dim=256,
                    input_length=30,
                    weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.1)))
model.add(Dense(10))
model.add(Dropout(0.35))
model.add(Dense(2, activation='softmax'))
optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=10)

pred = model.predict(X_dev)

print(f1_score(y_dev, pred))




