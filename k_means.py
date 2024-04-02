import numpy as np

class KMeans:
    def __init__(self, n_clusters=5, tolerance=0.07, max_iter=50):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter

    def fit(self, X):
        row_count, col_count = X.shape
        X_values = self.__get_values(X)
        X_labels = np.zeros(row_count)

        costs = np.zeros(1)
        all_clusterings = []

        for i in range(1):
            cluster_means = self.__initialize_means(X_values, self.n_clusters)
            for _ in range(self.max_iter):
                previous_means = np.copy(cluster_means)
                distances = self.__compute_distances(X_values, cluster_means, row_count)
                X_labels = self.__label_examples(distances)
                cluster_means = self.__compute_means(X_values, X_labels, col_count)
                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break

            X_values_with_labels = np.append(X_values, X_labels[:, np.newaxis], axis=1)
            all_clusterings.append((cluster_means, X_values_with_labels))
            costs[i] = self.__compute_cost(X_values, X_labels, cluster_means)

        best_clustering_index = costs.argmin() # задел на несколько запусков, у нас всего 1 эл-т в массиве, он и будет мин
        self.cost_ = costs[best_clustering_index] # задел на несколько запусков, у нас всего 1 эл-т в массиве, он и будет мин
        return all_clusterings[best_clustering_index]

    def var_part(self, X, n_clusters):
        """
        Method for previous initial centers and classters
        The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid.
        Since this is a measure of error, the objective of k-means is to try to minimize this value.

        Parameters:
        - X (array): input data
        - n_clusters (int): number of classters

        Returns:
        - initial_centers (array): previous centers
        """
        X_ = np.append(X, np.zeros(X.shape[0])[:, np.newaxis], axis=1)
        initial_centers = np.zeros((n_clusters, X.shape[1]))

        cluster_i = 1
        # оптимальная инициализация начальных центров и классов
        while cluster_i != n_clusters:
            within_clusters_sum_squares = np.zeros(cluster_i)
            for j in range(cluster_i):
                cluster_members = X_[X_[:, -1] == j]
                cluster_mean = cluster_members.mean(axis=0)
                within_clusters_sum_squares[j] = np.linalg.norm(cluster_members - cluster_mean, axis=1).sum()

            # Cluster which has max SSE
            max_sse_i = within_clusters_sum_squares.argmax()
            X_max_sse_i = X_[:, -1] == max_sse_i
            X_max_sse = X_[X_max_sse_i]

            variances, means = X_max_sse.var(axis=0), X_max_sse.mean(axis=0)
            max_variance_i = variances.argmax()
            max_variance_mean = means[max_variance_i]

            X_smaller_mean = X_max_sse[:, max_variance_i] <= max_variance_mean
            X_greater_mean = X_max_sse[:, max_variance_i] > max_variance_mean

            initial_centers[max_sse_i] = X_max_sse[X_smaller_mean].mean(axis=0)[:-1]
            initial_centers[cluster_i] = X_max_sse[X_greater_mean].mean(axis=0)[:-1]

            X_[(X_max_sse_i) & (X_[:, max_variance_i] <= max_variance_mean), -1] = cluster_i
            X_[(X_max_sse_i) & (X_[:, max_variance_i] > max_variance_mean), -1] = max_sse_i
            cluster_i += 1

        return initial_centers

    def __initialize_means(self, X, n_clusters):
        try:
            means = self.var_part(X, n_clusters)
            return means
        except:
            pass


    def __compute_distances(self, X, cluster_means, row_count):
        """
        Method for previous initial centers of classter
        The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid.
        Since this is a measure of error, the objective of k-means is to try to minimize this value.

        Parameters:
        - X (array): input data
        - cluster_means (array): centers
        - row_count (int) : count of row in data

        Returns:
        - distances (array): array of distances from centers to each classter example
        """
        # рассчет растояния от каждой точки в классе до центроида
        distances = np.zeros((row_count, self.n_clusters))
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis=1) # корень из сум кв разн соотв коорд

        return distances

    def __label_examples(self, distances):
        return distances.argmin(axis=1) # возвращает координаты точки с наименьшими расстояниями (новый центр)

    def __compute_means(self, X, labels, col_count):
        # разбиваем все точки на кластеры по признаку близости к новому центроиду
        """
        Compute new classter from all data elements
        Parameters:
        - X (array): input data
        - labels (array): new centers
        - col_count (int) : count of column in data
        Returns:
        - cluster_means (array): array of elements in each classter
        """
        cluster_means = np.zeros((self.n_clusters, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            cluster_elements = X[labels == cluster_mean_index]
            if len(cluster_elements):
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis=0)

        return cluster_means

    def __compute_cost(self, X, labels, cluster_means):
        """
        Parameters:
        - X (array): input data
        - labels (array): new centers
        - cluster_means (array): array of elements in each classter
        Returns:
        - cost (int): sum of each classter sum of all dictance in
        """
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X[labels == cluster_mean_index]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis=1).sum() #  сумма всех растояний внутри класстеров
        return cost

    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X)