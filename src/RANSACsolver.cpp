//
// Created by Johannes Marold on 18.06.24.
//

#include "../include/RANSACsolver.h"

// Custom equality comparator for std::pair
template<typename T1, typename T2>
struct PairEqual {
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
        return std::tie(lhs.first, lhs.second) == std::tie(rhs.first, rhs.second) ||
               std::tie(lhs.first, lhs.second) == std::tie(rhs.second, rhs.first);
    }
};

// Custom hash function for std::pair
template<typename T1, typename T2>
struct PairHash {
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        // Combine hash of first and second using bitwise XOR
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

// Custom hash function for unordered_set<int>
struct UnorderedSetHash {
    std::size_t operator()(const std::unordered_set<int>& uSet) const {
        std::size_t hash = 0;
        for (const int& elem : uSet) {
            // Combine the hash of each element
            hash ^= std::hash<int>()(elem) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Custom equality function for unordered_set<int>
struct UnorderedSetEqual {
    bool operator()(const std::unordered_set<int>& uSet1, const std::unordered_set<int>& uSet2) const {
        return uSet1 == uSet2;  // Compare sets directly
    }
};

/**
 * Function to compute the binomial coefficient
 * source: https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
 */
static unsigned long binomialCoefficient(const int n, const int k) {
    std::vector<unsigned long> aSolutions(k);
    aSolutions[0] = n - k + 1;

    for (int i = 1; i < k; ++i) {
        aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
    }

    return aSolutions[k - 1];
}

/**
 * Calculate Euclidean distance from a point to a flat in Normal Form
 *
 * @param normal Normal vector to the flat (dimensionality d)
 * @param c Offset scalar c in the equation n^T * x = c
 * @param singlePoint The point to which the distance is calculated (dimensionality d)
 *
 * @return distance as a float
 */
static float distancePointFlatNF(const MatrixXd& N, VectorXd c, const VectorXd& singlePoint) {

    // Ensure the dimensions are consistent
    assert(N.rows() == singlePoint.size());
    assert(N.cols() == c.size());

    // Compute the projection of the point onto the orthogonal complement (spanned by the normal vectors)
    VectorXd projection = N.transpose() * singlePoint;

    // Compute the distance as the norm of the projection minus the offset
    VectorXd distanceVector = projection - c;

    // Return the norm of the distance vector
    return distanceVector.norm();
}

 std::pair<MatrixXd , VectorXd> RANSACsolver::makeNormalForm(const MatrixXd& pointMatrix, int k) {

    if (k < 1 || k > pointMatrix.cols()-1) {
        std::string errorMessage = "k not in valid range";
        throw std::runtime_error(errorMessage);
    }

    int dim = pointMatrix.cols();
    MatrixXd pointsToVectorMatrix(dim, k);

    // Form the matrix using vectors (p1 - p0), (p2 - p0), ..., (pk-1 - p0)
    for (int i = 1; i <= k; ++i) {
        pointsToVectorMatrix.col(i - 1) = pointMatrix.row(i) - pointMatrix.row(0);
    }

    // Perform QR decomposition
    HouseholderQR<MatrixXd> qr(pointsToVectorMatrix);
    MatrixXd Q = qr.householderQ();
    MatrixXd N = Q.rightCols(dim - k);

    VectorXd c = N.transpose() * VectorXd(pointMatrix.row(0));

    return {N, c};
}

/**
 * determine d new random Points of dimension d from the Point Cloud, which has not yet been selected
 * @param selectedPoints that have already been picked (as hash set)
 * @param N Point Cloud size
 * @param d dimensions of points
 *
 * @return indices of new random Points
 */
static std::unordered_set<int> randomPointsForFlat(std::unordered_set<std::unordered_set<int>, UnorderedSetHash, UnorderedSetEqual>& selectedPoints, int N, int d) {

    std::unordered_set<int> pickedPoints;
    // Create a random device and a random number generator
    std::random_device rd;  // Seed generator
    std::mt19937 generator(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> distribution(0, N - 1);

    do {
        pickedPoints.clear();
        while (int(pickedPoints.size()) < d) {
            int randInt = distribution(generator);
            pickedPoints.insert(randInt);
        }
    } while (selectedPoints.find(pickedPoints) != selectedPoints.end());

    selectedPoints.insert(pickedPoints);
    return pickedPoints;
}

/**
 * extract points based on indices
 * @param indices of points
 * @param points as matrix, where each row is a point
 *
 * @return extracted points as matrix
 */
static MatrixXd extractPoints(const MatrixXd& points, std::unordered_set<int> indices) {
    MatrixXd extractPoints(indices.size(), points.cols());
    int i = 0;
    for (const int index : indices) {
        extractPoints.row(i) = points.row(index);
        i++;
    }
    return extractPoints;
}

/**
 * helper function for calculating the median of input data
 */
double computeMedian(std::vector<double>& data) {
    if (data.empty()) {
        throw std::runtime_error("The input vector is empty.");
    }

    size_t size = data.size();
    std::sort(data.begin(), data.end());

    if (size % 2 == 0) {
        // Even number of elements: average the two middle values
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else {
        // Odd number of elements: return the middle value
        return data[size / 2];
    }
}

double RANSACsolver::calcMAD() const {

    RowVectorXd m(X.cols()+1);

    // X values
    for (int i = 0; i < X.cols(); ++i) {
        std::vector<double> values(X.col(i).data(), X.col(i).data() + X.col(i).size());

        // Step 1: Compute the median of the original vector
        double median = computeMedian(values);

        // Step 2: Compute the absolute deviations from the median
        std::vector<double> absoluteDeviations;
        for (double val: values) {
            absoluteDeviations.push_back(std::abs(val - median));
        }

        // Step 3: Compute the median of the absolute deviations from the median of the original data
        m(i) = computeMedian(absoluteDeviations);
    }

    // y value
    std::vector<double> values(y.data(), y.data() + y.size());

    // Step 1: Compute the median of the original vector
    double median = computeMedian(values);

    // Step 2: Compute the absolute deviations from the median
    std::vector<double> absoluteDeviations;
    for (double val: values) {
        absoluteDeviations.push_back(std::abs(val - median));
    }

    // Step 3: Compute the median of the absolute deviations
    m(m.size()-1) = computeMedian(values);

    MatrixXd designMatrix(X.rows(), X.cols()+1);
    designMatrix << X, y;

    MatrixXd deviations = designMatrix.rowwise() - m;
    std::vector<double> absoluteDeviationsAll;

    // Compute the row-wise Manhattan norm (L1 norm) or Euclidian norm (L2 norm)
    for (int i = 0; i < X.rows(); ++i) {
        absoluteDeviationsAll.push_back(deviations.row(i).norm()); // L2
        //absoluteDeviationsAll.push_back(deviations.row(i).cwiseAbs().sum()); // L1
    }

    return computeMedian(absoluteDeviationsAll);
}

std::pair<VectorXd, double> RANSACsolver::detectInlierOutlier(int k) {

    MatrixXd designMatrix(X.rows(), X.cols()+1);
    designMatrix << X, y;
    int d = designMatrix.cols();

    if (iterations > binomialCoefficient(designMatrix.rows(),d)) {
        // maximum iterations (n choose k)
        std::string errorMessage = "maximum iterations possible: " + std::to_string((binomialCoefficient(designMatrix.rows(),d)));
        throw std::runtime_error(errorMessage);
    }

    std::unordered_set<std::unordered_set<int>, UnorderedSetHash, UnorderedSetEqual> selectedPoints;
    std::unordered_set<int> indicesPickedPoints;
    std::vector<bool> bestInlier(designMatrix.rows(), false);
    MatrixXd pickedPoints;
    std::pair<VectorXd, double> bestFlat;
    float inlierSizeMaximum = 0;

    for (unsigned int i = 0; i < iterations; ++i) {
        int currentInlierSize = 0;
        std::vector<bool> currentInlier(designMatrix.rows(), false);
        indicesPickedPoints = randomPointsForFlat(selectedPoints, designMatrix.rows(), d);
        pickedPoints = extractPoints(designMatrix, indicesPickedPoints);
        auto [N, C] = makeNormalForm(pickedPoints, k);
        for (int j = 0; j < designMatrix.rows(); j++) {
            float distance = distancePointFlatNF(N, C, designMatrix.row(j));
            if (distance < threshold) {
                currentInlierSize ++;
                currentInlier[j] = true;
            }
        }
        if (currentInlierSize > inlierSizeMaximum) { // check if current Flat supports more Points than previous best
            bestFlat = {N.col(0), C[0]}; // use case for hyper plane as regressor, for medianSDF irrelevant
            bestInlier = currentInlier;
            inlierSizeMaximum = currentInlierSize;
        }
    }

    inlier = bestInlier;
    return bestFlat;
}

void RANSACsolver::solve() {

    // RANSAC as Regressor fits a model using a hyperplane
    auto [n, c] = detectInlierOutlier(X.cols());

    VectorXd nNormalized = n / n(n.size()-1);

    w = -nNormalized.head(nNormalized.size()-1);
    bias = c / n(n.size()-1);

    std::cout << "\tRANSAC weight values: ";
    for (Eigen::Index i = 0; i < w.size(); ++i) {
        std::cout << w[i];
        if (i < w.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ", and bias: " << bias << std::endl;
}

