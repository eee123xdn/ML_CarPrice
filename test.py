# loading packages
import matplotlib
import numpy as np
import pandas as pd
import datetime

# data visualization and missing values
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# stats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import mean_squared_error, r2_score

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

seed = 123  # 设置一个随机种子，不会每次随机抽样的结果不同

data = pd.read_csv("CarPrice_Assignment1.csv", na_values='?')  # 把表格或者数据里面为'?'，看做缺失
# 查看数据信息
print(data.columns)  # 查看每列列名
print('-----------------------------------------------')
print(data.dtypes)  # 查看字符类型
print('-----------------------------------------------')
print(data.shape)  # 查看数据有几行几列
print('-----------------------------------------------')
print(data.head(5))  # 查看前面5行内容
print('-----------------------------------------------')
print(data.describe())  # 数据描述，比如均值，标准差什么的

# 缺失值处理 （missingno缺失值可视化）
sns.set(style="ticks")  # 指定风格
msno.matrix(data)  # 画图
plt.show()

data = data.dropna(subset=['price', 'boreratio', 'stroke', 'peakrpm', 'horsepower', 'doornumber'])  # 对于缺失值少的几列，直接删掉缺失值
print('In total:', data.shape)

# 使用corr()计算数据的相关性，返回的仍是dataframe类型数据，可以直接引用
# 相关系数的取值范围为[-1, 1],当接近1时，表示两者具有强烈的正相关性，
# 比如‘s’和‘x’；当接近-1时，表示有强烈的的负相关性，比如‘s’和‘c’，
# 而若值接近0，则表示相关性很低
# cormatrix #查看结果,是一个对称矩阵
cormatrix = data.corr()

# 不同的展现格式
cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T  # 返回函数的上三角矩阵，把对角线上的置0，让他们不是最高的。
cormatrix = cormatrix.stack()  # 某一指标与其他指标的关系

# 找出前十个最相关的特征
cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
print(cormatrix.head(10))

data['volume'] = data.carwidth * data.carlength * data.carheight

data.drop(['carwidth', 'carlength', 'carheight',
           'curbweight', 'citympg'],
          axis=1,  # 1 for columns
          inplace=True)

corr_all = data.corr()
# 热力图展示
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask=mask,
            square=True, linewidths=.5, ax=ax, cmap="RdBu_r")
plt.show()

print('fueltype:', data['fueltype'].unique(), '\ndoors:', data['doornumber'].unique())
sns.lmplot(data, x='price', y='horsepower', hue='fueltype',
           col='fueltype', row='doornumber', palette='plasma',
           fit_reg=True)
plt.show()

target = data.price
# 特征数据
features = data.drop(columns=['price'])
# 数字类型的特征,连续型的进行标准化
num = ['symboling', 'volume', 'horsepower', 'wheelbase',
       'boreratio', 'stroke', 'compressionratio', 'peakrpm', 'enginesize', 'highwaympg']

# 标准化处理
standard_scaler = StandardScaler()
features[num] = standard_scaler.fit_transform(features[num])
print(features.shape)

## 需要进行one-hot编码的特征列
classes = ['CarName', 'fueltype', 'aspiration', 'doornumber',
           'carbody', 'drivewheel', 'enginelocation',
           'enginetype', 'cylindernumber', 'fuelsystem']

## 使用pandas的get_dummies进行one-hot编码
dummies = pd.get_dummies(features[classes])
print(dummies.columns)

## one-hot编码加工好的特征数据
features = features.join(dummies).drop(classes, axis=1)
print(features.shape)
print(features.head())

# 按照30%划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.3,
                                                    random_state=seed)
print("Train", X_train.shape, "and test", X_test.shape)

# Lasso回归,基于线性回归的基础上，多加了一个绝对值想来惩罚过大的系数，即+\lambda*|w|,比如w=[1/4,1/4,1/4,1/4]和w=[1,0,0,0]的模型，前者更优
# lassocv：交叉验证模型，

# 交叉验证cross validation
lassocv = LassoCV(cv=10, random_state=seed)
# 制定模型，将训练集平均切10分，9份用来做训练，1份用来做验证，可设置alphas=[]是多少（序列格
# 式），默认不设置则找适合训练集最优alpha
lassocv.fit(features, target)
lassocv_score = lassocv.score(features, target)  # 测试模型,返回r^2值
lassocv_alpha = lassocv.alpha_

plt.figure(figsize=(10, 4))
plt.plot(lassocv_alpha, lassocv_score, '-ko')
plt.axhline(lassocv_score, color='c')
plt.xlabel(r'$\alpha$')
plt.ylabel('CV Score')
plt.xscale('log', base=2)
sns.despine(offset=15)
plt.show()
# 将不同\alpha的效果与最佳alpha的效果对比
print('CV results:', lassocv_score, lassocv_alpha)

# lassocv coefficients
coefs = pd.Series(lassocv.coef_, index=features.columns)

# prints out the number of picked/eliminated features
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
      str(sum(coefs == 0)) + " features.")

# 展示前5个和后5个
coefs = pd.concat([coefs.sort_values().head(5), coefs.sort_values().tail(5)])

plt.figure(figsize=(10, 4))
coefs.plot(kind="barh", color='c')
plt.title("Coefficients in the Lasso Model")
plt.show()

# 用上面算出来的\alpha去训练模型
model_l1 = LassoCV(alphas=[lassocv.alpha_], cv=10, random_state=seed).fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)
# 输出测试的结果
print(model_l1.score(X_test, y_test))

# 用上面训练的模型查看测试集真实值和预测值的差异,残差图展示
plt.rcParams['figure.figsize'] = (6.0, 6.0)
# 构造pandas 数据库。preds：预测值，true：真实值，residuals：真实值-预测值
preds = pd.DataFrame({"preds": y_pred_l1, "true": y_test})
preds["residuals"] = preds["true"] - preds["preds"]
# 可视化 {preds：预测值 }和 {residuals：真实值-预测值 }之间的关系
sns.scatterplot(x='preds', y="residuals", data=preds)
plt.show()


# 计算指标：MSE和R2
def MSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse


def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    print('R2: %2.3f' % r2)
    return r2

MSE(y_test, y_pred_l1)
R2(y_test, y_pred_l1)

#结果预测
# predictions
d = {'true' : list(y_test),
     'predicted' : pd.Series(y_pred_l1)
    }

print(pd.DataFrame(d).head())
