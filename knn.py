import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')



def l2_norm(vector1, vector2):
    """
    Input:
        vector1, vector2: numpy arrays having the same shape
    Output:
        distance: A float number represents for L2 norm between 2 vectors
    """
    vector1 = vector1.reshape(vector2.shape)
    assert vector1.shape == vector2.shape
    distance = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return distance


def knn_classifier(train_set, y, test_sample, K):
    """
    Input:
        train_set: Training data
        y: ground truth values of training data (label)
        test_sample: A test instance
        K: The number of nearest neighbors
    Output:
        (predicted_label, neighbors)
    """
    n_samples = train_set.shape[0]
    distances = []
    for i in range(n_samples):
        distance = l2_norm(test_sample, train_set.loc[i, 'SepalLengthCm':'PetalWidthCm'])
        distances.append((i, distance))

    sorted_distances = sorted(distances, key=lambda x: x[1]) # ascending

    neighbors = []
    for i in range(K):
        neighbors.append(sorted_distances[i][0])

    n_votes = {}
    for i in range(K):
        neighbor_label = y[neighbors[i]]
        n_votes[neighbor_label] = n_votes.get(neighbor_label, 0) + 1
    
    sorted_votes = sorted(n_votes.items(), key=lambda x: x[1], reverse=True) # descending

    predicted_label = sorted_votes[0][0]
    
    return predicted_label, neighbors


if __name__ == "__main__":
    # Step 1: Loading data
    df = pd.read_csv('iris.csv')
    print(df.head())
    
    # Step 2: Initializing data and K
    # Prepare training dataset, ground truth label, a test sample
    train_set = df.loc[:, 'SepalLengthCm':'PetalWidthCm']
    y = df.loc[:, 'Species']
    test_sample = np.array([[7.2, 3.6, 5.1, 2.5]])
    K = 3 # K nearest neighbors

    # Step 3: Run the KNN model
    predicted_name, neighbors = knn_classifier(train_set, y, test_sample, K)

    print('\n3 Nearest Neighbour')
    # Predicted name
    print('\nPredicted Name:', predicted_name)

    # Nearest neighbors
    print('Nearest Neighbour of the test sample = ', neighbors)

    # Compare to KNN in the scikit-learn library
    print('Compare to scikit-learn KNN model')
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(df.loc[:,'SepalLengthCm':'PetalWidthCm'], df['Species'])

    print('\nScikit-learn predicted name:', neigh.predict(test_sample))
    print('Scikit-learn nearest neighbors = ', neigh.kneighbors(test_sample)[1])




