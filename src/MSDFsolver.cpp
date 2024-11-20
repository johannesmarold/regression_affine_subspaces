//
// Created by Johannes Marold on 26.08.24.
//

#include "../include/MSDFsolver.h"

/**
 * shuffle indices of points for randomly picking points
 * @param pointsSize
 *
 * @return shuffled indices
 */
static std::vector<int> shuffleIndices(int pointsSize) {

    std::vector<int> shuffledIndices(pointsSize);
    std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0); // Fill with 0, 1, 2, ..., points.size()-1

    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), generator);

    return shuffledIndices;
}

/**
 * function assigns for a given list of points randomly point groups and returns the list of grouped points in normal form
 * @param points
 * @param shuffledIndices randomly shuffled indices of points
 * @param allFlatsPF list contains all flats in parameter form
 * @param k dimension of flats
 */
static void listOfFlatsNF(const MatrixXd& points, const std::vector<int>& shuffledIndices, std::vector<std::pair<MatrixXd, VectorXd>>& allFlatsNF, int k) {
    // k+1 points for k-Flat necessary
    int kPoints = k+1;
    for (int i = 0; i < points.rows(); i+=kPoints) {
        MatrixXd pointsForFlat(kPoints, points.cols());
        int pointsForFlatIndex = 0;
        for (int j = i; j < i+kPoints; ++j) {
            pointsForFlat.row(pointsForFlatIndex) = points.row(shuffledIndices[(j) % (points.rows()-1)]);
            pointsForFlatIndex++;
        }
        std::pair<MatrixXd, VectorXd> flatInNF = RANSACsolver::makeNormalForm(pointsForFlat, k);
        allFlatsNF.push_back(flatInNF);
    }
}

/**
 * function that assigns the inlier-inlier and outlier-outlier pairs of the sorted RANSAC results
 * approach: 1. sort two Lists of Points, one Inlier, one Outlier-List
 *           2. group together k+1 points for each List to form a k-flat
 *           3. return List of k-flats (first all Inlier-flats, afterwards appending all Outlier-flats)
 * @param points
 * @param inlier boolean vector for detecting which points are inlier
 * @param kMedianSDF k for k-flats
 * @param kFlatAmount amount of flats = kFlatAmount * (points.rows()/kMedianSDF)
 *
 * @return List of all pairs as Flats represented in normal Form
 */
static std::vector<std::pair<MatrixXd, VectorXd>> assignFlatsNF(const MatrixXd& points, std::vector<bool> inlier, int kMedianSDF, int kFlatAmount) {

    int inlierSize = std::count(inlier.begin(), inlier.end(), true);

    MatrixXd inlierPoints(inlierSize, points.cols());
    int inlierPointsIndex = 0, outlierPointsIndex = 0;
    MatrixXd outlierPoints(points.rows()-inlierSize, points.cols());

    for (int i = 0; i < points.rows(); ++i) {
        if (inlier[i]) {
            inlierPoints.row(inlierPointsIndex) = points.row(i);
            inlierPointsIndex++;
        }
        else {
            outlierPoints.row(outlierPointsIndex) = points.row(i);
            outlierPointsIndex++;
        }
    }

    std::vector<std::pair<MatrixXd, VectorXd>> allFlatsInNF;

    for (int i = 0; i < kFlatAmount; ++i) {
        // Shuffle the points to ensure randomness
        std::vector<int> shuffledInlierIndices = shuffleIndices(inlierPoints.rows());
        std::vector<int> shuffledOutlierIndices = shuffleIndices(outlierPoints.rows());

        listOfFlatsNF(inlierPoints, shuffledInlierIndices, allFlatsInNF, kMedianSDF);
        listOfFlatsNF(outlierPoints, shuffledOutlierIndices, allFlatsInNF, kMedianSDF);
    }

    return allFlatsInNF;
}

/**
 * function converts flat in normal form to SDF (squared distance function)
 * k' = d-k
 * @param N k'xd matrix containing the normal vectors of the flat
 * @param c k' vector representing the displacement of the flat
 *
 * @return pair consisting of Q (dxd matrix that represents the flat's dimension and orientation) and
 *         r (d vector representing the flat's shift)
 */
static std::pair<MatrixXd, VectorXd> normalFormToSDF(const Eigen::MatrixXd &N, const Eigen::VectorXd &c) {

    Eigen::MatrixXd Q;
    Eigen::VectorXd r;
    int d = N.rows();
    int k_dash = N.cols();

    Q = N * N.transpose();
    r = -N*c;

    return {Q, r};
}

/**
 * function applies normalFormToSDF() for a list of flats in normal form
 *
 * @param Ncs list of flats in normal form
 *
 * @return list of flats in SDF as pair containing Q and r
 */
static std::pair<std::vector<MatrixXd>, std::vector<VectorXd>> normalFormsToSDFs(const std::vector<std::pair<MatrixXd, VectorXd>>& Ncs) {

    std::vector<MatrixXd> Qs;
    std::vector<VectorXd> rs;
    for (size_t i = 0; i < Ncs.size(); ++i) {
        std::pair<MatrixXd, VectorXd> singleSDF = normalFormToSDF(Ncs[i].first, Ncs[i].second);
        Qs.push_back(singleSDF.first);
        rs.push_back(singleSDF.second);
    }
    return {Qs, rs};
}

/**
 * Matrix Version
 * iteratively updates the median estimation by weighted averaging of the input matrices
 * coefs are updated based on the the inverse of the distance (norm) between curr and each matrix in mats
 *
 * @param mats list of matrices, for which the median is estimated
 * @param errorTolerance until the median converges
 * @param maxIter until the function determines
 *
 * @return the median as matrix
 */
static MatrixXd computeMedian(const std::vector<MatrixXd>& mats, double errorTolerance=1e-12, int maxIter=1000) {

    MatrixXd out;

    int n = mats.size();
    int r = mats[0].rows();
    int c = mats[0].cols();

    MatrixXd prev(r, c);
    MatrixXd curr(r, c);

    curr.setOnes();
    prev.setZero();

    Eigen::ArrayXd coefs = Eigen::ArrayXd::Constant(n, 1. / (double)n);
    int it = 0;

    while (it < maxIter && (curr - prev).norm() > errorTolerance)
    {
        prev = curr;
        curr.setZero();
        for (int i = 0; i < n; i++)
        {
            curr += coefs(i) * mats[i];
        }

        for (int i = 0; i < n; i++)
        {
            coefs(i) = (curr - mats[i]).norm();
        }
        coefs = coefs.cwiseInverse();
        coefs /= coefs.sum();

        it++;
    }

    out = curr;

    return out;
}

/**
 * Vector Version
 * iteratively updates the median estimation by weighted averaging of the input vectors
 * coefs are updated based on the the inverse of the distance (norm) between curr and each vector in mats
 *
 * @param vecs list of vectors, for which the median is estimated
 * @param errorTolerance until the median converges
 * @param maxIter until the function determines
 *
 * @return the median as vector
 */
static VectorXd computeMedian(const std::vector<VectorXd>& vecs, double errorTolerance=1e-12, int maxIter=1000) {

    VectorXd out;

    int n = vecs.size();
    int r = vecs[0].rows();
    int c = vecs[0].cols();

    VectorXd prev(r, c);
    VectorXd curr(r, c);

    curr.setOnes();
    prev.setZero();

    Eigen::ArrayXd coefs = Eigen::ArrayXd::Constant(n, 1. / (double)n);
    int it = 0;

    while (it < maxIter && (curr - prev).norm() > errorTolerance)
    {
        prev = curr;
        curr.setZero();
        for (int i = 0; i < n; i++)
        {
            curr += coefs(i) * vecs[i];
        }

        for (int i = 0; i < n; i++)
        {
            coefs(i) = (curr - vecs[i]).norm();
        }
        coefs = coefs.cwiseInverse();
        coefs /= coefs.sum();

        it++;
    }

    out = curr;

    return out;
}

/**
 * function to calculate the medianSDF Flat for a given list of flats represented in normal form
 * and extracting weight vector plus bias
 *
 * @param Qs list of Qs of the SDF form of a flat
 * @param rs list of rs of the SDF form of a flat
 *
 * @return bias plus weight vector together as VectorXd
 */
static std::pair<VectorXd, float> medianSDF(const std::vector<Eigen::MatrixXd> &Qs, const std::vector<Eigen::VectorXd> &rs) {

    Eigen::MatrixXd Q_star = computeMedian( Qs);
    Eigen::VectorXd r_star = computeMedian(rs);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>
            evd(Q_star);

    auto U = evd.eigenvectors();
    // Pseudo-inverse of the diagonal matrix Lambda
    auto Lp = evd.eigenvalues().unaryExpr([](const double &x)
                                          { return abs(x) < 1e-10 ? x : 1.0 / x; })
            .asDiagonal();

    Eigen::MatrixXd Qp = U * Lp * U.transpose();

    // Extract eigenvector (normal vector for the normal form)
    VectorXd n = U.rightCols(1);

    // Calculate the offset c as -n^T*Qp*r^*
    double c = -n.transpose() * Qp * r_star;

    VectorXd nNormalized = n / n(n.size()-1);
    float bias = c / n(n.size()-1);
    VectorXd w = -nNormalized.head(nNormalized.size()-1);

    return {w, bias};
}

void MSDFsolver::solve() {

    if (inlier.empty()) {
        detectInlierOutlier(kPreProcess);
    }

    MatrixXd designMatrix(X.rows(), X.cols()+1);
    designMatrix << X, y;

    std::vector<std::pair<MatrixXd, VectorXd>> groupedFlatsNF = assignFlatsNF(designMatrix, inlier, kMedianSDF, kFlatAmount);
    auto [Qs, rs] = normalFormsToSDFs(groupedFlatsNF);
    auto [wNew, biasNew] = medianSDF(Qs, rs);

    bias = biasNew;
    w = wNew;

    std::cout << "\tMedianSDF weight values: ";
    for (Eigen::Index i = 0; i < w.size(); ++i) {
        std::cout << w[i];
        if (i < w.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ", and bias: " << bias << std::endl;
}