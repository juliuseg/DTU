import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
import numpy as np
import xlrd
import pandas as pd
import scipy.stats as stats

filename = 'PowerPlant.xls'
doc = xlrd.open_workbook(filename).sheet_by_index(0)

attributeNames = doc.row_values(0, 0, 5)
print(attributeNames)


X = np.empty((9567, 5)) 
for i, col_id in enumerate(range(0, 5)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 9568))




N, M = X.shape


Y = X - np.ones((N, 1)) * X.mean(axis=0)

U, S, V = svd(Y, full_matrices=False)

rho = (S * S) / (S * S).sum()



df = pd.DataFrame(X, columns=attributeNames)

def QQplots():
    # Create QQ plots for each column in X
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(X.shape[1]):
        stats.probplot(X[:, i], dist="norm", plot=axs[i])
        axs[i].set_title(f'QQ Plot for {attributeNames[i]}')

    plt.tight_layout()
    plt.show()

def PCAexplanations():
    pcs = [0, 1]
    legendStrs = ["PC" + str(e + 1) for e in pcs]
    c = ["r", "g", "b"]
    bw = 0.2
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, attributeNames)
    plt.xlabel("Attributes")
    plt.ylabel("Component coefficients")
    plt.legend(legendStrs)
    plt.grid()
    plt.title("Powerplant: PCA Component Coefficients")
    plt.show()


def describeData():
    print("\nData describtion")
    s = df.describe()
    print(s)

    print("\nCorralation Matrix\n")
    correlation_matrix_labeled = df.corr()
    print(correlation_matrix_labeled)

    



def explained():
    threshold1 = 0.90
    threshold2 = 0.95
    print(rho)
    print(np.cumsum(rho))

    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([1, len(rho)], [threshold1, threshold1], "k--")
    plt.plot([1, len(rho)], [threshold2, threshold2], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.show()

def scatter():
    Z = Y.dot(V.T)

    PCA1 = Z[:, 0]
    PCA2 = Z[:, 1]

    # Plot
    plt.figure()
    plt.scatter(PCA1, PCA2)
    plt.title('PCA Scatter Plot')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)
    plt.show()

def boxplot(data):

    means = data.mean(axis=0)
    std_devs = data.std(axis=0)

    print(means)
    print(std_devs)
    print(data)

    X_standardized = (data - means) / std_devs
    plt.figure(figsize=(12, 8))  

    plt.boxplot(X_standardized, notch=False, sym='o', vert=True, whis=1.5, patch_artist=False, showmeans=False)

    plt.xticks(ticks=np.arange(1, len(attributeNames) + 1), labels=attributeNames)

    plt.title('Boxplot of Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Values')

    plt.grid(True)
    plt.show()

def scatterMatrix(data, figsize=(8, 8)):
    num_vars = data.shape[1]
    fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=figsize)
    
    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]
            if i == j:  
                ax.hist(data[:, i])
            else:  
                ax.scatter(data[:, j], data[:, i], s=1)
            
            if i == num_vars - 1:
                ax.set_xlabel(attributeNames[j])
            if j == 0:
                ax.set_ylabel(attributeNames[i])
            
            if i < num_vars - 1:
                ax.xaxis.set_ticklabels([])
            if j > 0:
                ax.yaxis.set_ticklabels([])
    
    plt.tight_layout()
    plt.show()

def scatterMatrixPower(data, figsize=(8, 8)):
    num_vars = 4
    fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=(figsize[0] + 2, figsize[1]))
    
    color = data[:, num_vars]
    min_color, max_color = np.min(color), np.max(color)
    color_normalized = (color - min_color) / (max_color - min_color)
    
    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]
            
            scatter = ax.scatter(data[:, j], data[:, i], c=color_normalized, cmap='coolwarm', s=10)
            
            if attributeNames and i == num_vars - 1:
                ax.set_xlabel(attributeNames[j])
            if attributeNames and j == 0:
                ax.set_ylabel(attributeNames[i])
            
            if i < num_vars - 1:
                ax.xaxis.set_ticklabels([])
            if j > 0:
                ax.yaxis.set_ticklabels([])
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Energy output')
    
    ticks = np.linspace(0, 1, num=5)  
    tick_labels = min_color + ticks * (max_color - min_color)  
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{label:.2f}' for label in tick_labels])  
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()



#describeData()
#scatter()
#boxplot(X)
#scatterMatrix(X)
#scatterMatrixPower(X)
#explained()
#PCAexplanations()
#dataOnPCs()
QQplots()