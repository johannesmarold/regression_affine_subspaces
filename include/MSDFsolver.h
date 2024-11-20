//
// Created by Johannes Marold on 26.08.24.
//

#ifndef BACHELOR_JOHANNES_MSDFSOLVER_H
#define BACHELOR_JOHANNES_MSDFSOLVER_H

#include "RANSACsolver.h"

class MSDFsolver : public RANSACsolver {
public:
    MSDFsolver(MatrixXd X, VectorXd y, int iterations, int kPreprocess, int kMedianSDF, float threshold=0.0, int kFlatAmount=1) :
            RANSACsolver(X, y, iterations, threshold),
            kPreProcess(kPreprocess), kMedianSDF(kMedianSDF), kFlatAmount(kFlatAmount)
            {
                if (kFlatAmount <= 0) {
                    throw std::invalid_argument("Value must be integer greater than 0.");
                }
            }

    int getKPreProcess() const {
        return kPreProcess;
    }

    int getKMedianSDF() const {
        return kMedianSDF;
    }

    /**
     * algorithm to calculate weight vector and bias based on new medianSDF approach
     * approach: 1. separate points in inlier and outlier using RANSAC
     *           2. build k-flats within each group (in normal form)
     *           3. convert normal form to SDF
     *           4. estimate the median as SDF of all given k-flats in SDF using Weiszfeld algorithm
     *           5. project median estimation into hyperplane-space
     *           6. extract weight vector and bias
     */
    virtual void solve();

private:
    /**
     * dimension of k-flat, that is used for detecting inlier and outlier with RANSAC
     */
    int kPreProcess;

    /**
     * dimension of k-flats, that are used for fitting a hyperplane through via MedianSDF
     */
    int kMedianSDF;

    /**
     * Multiple of number of data points divided by points required for k-Flat
     */
    int kFlatAmount;
};

#endif //BACHELOR_JOHANNES_MSDFSOLVER_H
