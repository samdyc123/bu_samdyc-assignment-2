let data = [];
let initMethod = 'random';
let nClusters = 3;
let kmeansState = null;  // 用于存储KMeans的当前状态
let manualCentroids = [];

document.getElementById('init-method').addEventListener('change', function() {
    initMethod = this.value;
    toggleCentroidInput();
});

function generateData() {
    fetch('/generate_data?num_points=100')
        .then(response => response.json())
        .then(jsonData => {
            data = jsonData;
            drawChart(data);
        });
}

function drawChart(data) {
    d3.select("#chart").selectAll("*").remove();

    const svg = d3.select("#chart").append("svg")
        .attr("width", 600)
        .attr("height", 400);

    svg.selectAll("circle")
        .data(data)
        .enter().append("circle")
        .attr("cx", d => d[0] * 600)
        .attr("cy", d => d[1] * 400)
        .attr("r", 5);

    if (initMethod === 'manual') {
        svg.on("click", function(event) {
            const [x, y] = d3.pointer(event);
            manualCentroids.push([x / 600, y / 400]);
            svg.append("rect")
                .attr("x", x - 5)
                .attr("y", y - 5)
                .attr("width", 10)
                .attr("height", 10)
                .attr("fill", "red");
        });
    }
}

function runKMeans() {
    const payload = {
        data: data,
        init_method: initMethod,
        n_clusters: nClusters,
    };

    if (initMethod === 'manual') {
        payload.centroids = manualCentroids;
    }

    fetch('/kmeans', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
    .then(response => response.json())
    .then(result => {
        const centroids = result.centroids;
        const labels = result.labels;
        drawClusters(data, centroids, labels);
    });
}

function drawClusters(data, centroids, labels) {
    d3.select("#chart").selectAll("*").remove();

    const svg = d3.select("#chart").append("svg")
        .attr("width", 600)
        .attr("height", 400);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    svg.selectAll("circle")
        .data(data)
        .enter().append("circle")
        .attr("cx", d => d[0] * 600)
        .attr("cy", d => d[1] * 400)
        .attr("r", 5)
        .attr("fill", (d, i) => color(labels[i]));

    svg.selectAll("rect")
        .data(centroids)
        .enter().append("rect")
        .attr("x", d => d[0] * 600 - 5)
        .attr("y", d => d[1] * 400 - 5)
        .attr("width", 10)
        .attr("height", 10)
        .attr("fill", "black");
}

// 模拟逐步KMeans聚类过程
function stepKMeans() {
    if (!kmeansState) {
        // 初始化KMeans状态
        kmeansState = {
            step: 0,
            centroids: initializeCentroids(initMethod, nClusters, data),
            labels: Array(data.length).fill(-1)
        };
    }

    if (kmeansState.step % 2 === 0) {
        // 分配数据点到最近的质心
        kmeansState.labels = assignLabels(kmeansState.centroids, data);
    } else {
        // 重新计算质心位置
        kmeansState.centroids = recomputeCentroids(kmeansState.labels, data, nClusters);
    }

    kmeansState.step++;
    drawClusters(data, kmeansState.centroids, kmeansState.labels);
}

function convergeKMeans() {
    while (!isConverged(kmeansState)) {
        stepKMeans();
    }
}

function isConverged(state) {
    if (!state) return false;
    // Check if the centroids have stopped moving
    const newCentroids = recomputeCentroids(state.labels, data, nClusters);
    return JSON.stringify(newCentroids) === JSON.stringify(state.centroids);
}

function resetKMeans() {
    kmeansState = null;
    manualCentroids = [];
    drawChart(data);
}

// 初始化质心
function initializeCentroids(method, k, data) {
    let centroids = [];
    if (method === 'random') {
        for (let i = 0; i < k; i++) {
            centroids.push(data[Math.floor(Math.random() * data.length)]);
        }
    } else if (method === 'k-means++') {
        centroids = [data[Math.floor(Math.random() * data.length)]];
        while (centroids.length < k) {
            let distances = data.map(point => Math.min(...centroids.map(c => euclideanDistance(point, c))));
            let sumDistances = distances.reduce((a, b) => a + b, 0);
            let r = Math.random() * sumDistances;
            let cumulative = 0;
            for (let i = 0; i < distances.length; i++) {
                cumulative += distances[i];
                if (cumulative >= r) {
                    centroids.push(data[i]);
                    break;
                }
            }
        }
    } else if (method === 'farthest') {
        centroids = [data[Math.floor(Math.random() * data.length)]];
        while (centroids.length < k) {
            let maxDist = 0;
            let farthestPoint = null;
            for (let point of data) {
                let minDist = Math.min(...centroids.map(c => euclideanDistance(point, c)));
                if (minDist > maxDist) {
                    maxDist = minDist;
                    farthestPoint = point;
                }
            }
            centroids.push(farthestPoint);
        }
    } else if (method === 'manual') {
        return manualCentroids;
    }
    return centroids;
}

// 分配标签
function assignLabels(centroids, data) {
    return data.map(point => {
        let distances = centroids.map(c => euclideanDistance(point, c));
        return distances.indexOf(Math.min(...distances));
    });
}

// 重新计算质心
function recomputeCentroids(labels, data, k) {
    let newCentroids = Array(k).fill(null).map(() => [0, 0]);
    let counts = Array(k).fill(0);
    for (let i = 0; i < data.length; i++) {
        let clusterIndex = labels[i];
        newCentroids[clusterIndex][0] += data[i][0];
        newCentroids[clusterIndex][1] += data[i][1];
        counts[clusterIndex]++;
    }
    return newCentroids.map((sum, i) => [sum[0] / counts[i], sum[1] / counts[i]]);
}

// 计算欧几里得距离
function euclideanDistance(a, b) {
    return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
}

function toggleCentroidInput() {
    const initMethod = document.getElementById('init-method').value;
    const manualCentroidsDiv = document.getElementById('manual-centroids');
    if (initMethod === 'manual') {
        manualCentroidsDiv.style.display = 'block';
    } else {
        manualCentroidsDiv.style.display = 'none';
    }
}
