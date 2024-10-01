from flask import Flask, request, jsonify, render_template
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_data', methods=['GET'])
def generate_data():
    num_points = int(request.args.get('num_points', 100))
    data = np.random.rand(num_points, 2).tolist()
    return jsonify(data)


@app.route('/kmeans', methods=['POST'])
def kmeans():
    data = request.json['data']
    init_method = request.json['init_method']
    n_clusters = int(request.json['n_clusters'])

    if not data or len(data) == 0:
        return jsonify({'error': 'No data provided'}), 400

    data = np.array(data)

    if init_method == 'manual':
        centroids = request.json.get('centroids')
        if not centroids or len(centroids) == 0:
            return jsonify({'error': 'No centroids provided for manual initialization'}), 400
        centroids = np.array(centroids)
        
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        
        n_clusters = len(centroids)
        print(f"Number of clusters (manual): {n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    
    elif init_method == 'farthest':
        centroids = initialize_farthest_centroids(data, n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, n_init=10)
    
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_.tolist()
    labels = kmeans.labels_.tolist()
    
    return jsonify({'centroids': centroids, 'labels': labels})


def initialize_farthest_centroids(data, k):
    centroids = [data[np.random.choice(len(data))]]
    while len(centroids) < k:
        max_dist = 0
        farthest_point = None
        for point in data:
            min_dist = min(np.linalg.norm(point - np.array(c)) for c in centroids)
            if min_dist > max_dist:
                max_dist = min_dist
                farthest_point = point
        centroids.append(farthest_point)
    return np.array(centroids)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
