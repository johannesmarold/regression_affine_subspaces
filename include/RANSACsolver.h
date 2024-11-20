//
// Created by Johannes Marold on 18.06.24.
//

#ifndef BACHELOR_JOHANNES_RANSACSOLVER_H
#define BACHELOR_JOHANNES_RANSACSOLVER_H

#include "solver.h"

class RANSACsolver : public solver {
public:
    RANSACsolver(MatrixXd X, VectorXd y, int iterations, float threshold=0.0) :
            solver(X, y),
            iterations(iterations), threshold(threshold)
            {
                if (threshold == 0) {
                    this->threshold = calcMAD();
                }
            }

    int getIterations() const {
        return iterations;
    }

    float getThreshold() const {
        return threshold;
    }

    std::vector<bool> getInlier() const {
        return inlier;
    }

    /**
     * Function that creates the normal form of a k-flat from k+1 points
     * Normal form: N^T * x = c
     * Where N is a normal matrix orthogonal to the flat, and c is a offset vector
     * @param pointMatrix as matrix, where each row is a point
     * @param k dimension of flat
     * @return pair of matrix N (dimensionality d x k') and offset vector c (dimensionality k'), where k' = d-k
     */
    static std::pair<MatrixXd , VectorXd> makeNormalForm(const MatrixXd& points, int k);

    /**
     * function that detects Inlier and Outlier using RANSAC algorithm
     * approach:
     *  1. randomly sample d points from Point Cloud to fit a hyperplane
     *  2. calculate distance for every point from Point Cloud to the hyperplane
     *  3. distinguish every point in inlier and outlier according to distance larger or smaller than threshold
     *  4. repeat steps 1-3 until maximum iterations are reached, keep hyperplane, that supports the most inlier
     * parameters used:
     *  - iterations: how often should step 1-3 be repeated (depends on minimum size of points to determine line and size of outliers within data)
     *  - threshold: defines, which points are inlier and outlier of the current regression line
     *               how to choose: - data set driven like median NearestNeighborDistance or stddev
     *                              - model evaluation like grid search
     * @param k dimension of fitted model (aka fitted flat)
     * @return flat in normal form
     */
    virtual std::pair<VectorXd, double> detectInlierOutlier(int k);

    /**
     * solver for RANSAC using detectInlierOutlier function to detect best hyperplane
     */
    virtual void solve();

protected:
    /**
     * function to calculate a multivariate median absolute deviation of X and target value y
     * used as threshold if it is not specified
     * approach: MAD of the absolute deviation from each point to the MAD in each dimension
     *           1. compute MAD for each dimension
     *           2. compute absolute deviation of data from MAD in each dimension
     *           3. compute L1 or L2 norm of absolute deviations per dimension
     *           4. compute median of norms
     */
    double calcMAD() const;

    /**
     * amount of iterations used for searching the optimal regression line
     */
    unsigned int iterations;

    /**
     * threshold for distinguishing inlier and outlier
     * if not specified during initialization, MAD is used
     */
    float threshold;

    /**
     * vector of booleans says if point at index is inlier of current model or not
     */
    std::vector<bool> inlier;
};

#endif //BACHELOR_JOHANNES_RANSACSOLVER_H
