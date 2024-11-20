//
// Created by Johannes Marold on 21.05.24.
//

#ifndef BACHELOR_JOHANNES_SOLVER_H
#define BACHELOR_JOHANNES_SOLVER_H

#include "Eigen/Dense"
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "polyscope/combining_hash_functions.h"
#include "polyscope/messages.h"

#include "polyscope/file_helpers.h"
#include "polyscope/point_cloud.h"
#include "polyscope//curve_network.h"

using namespace Eigen;

class solver {

public:
    solver(MatrixXd X, VectorXd y)
    :X(X), y(y)
    {
    }

    MatrixXd getX() const {
        return X;
    }

    VectorXd getY() const {
        return y;
    }

    VectorXd getW() const {
        return w;
    }

    float getBias() const {
        return bias;
    }

    VectorXd predict(MatrixXd Xpred);

private:

protected:
/**
 * matrix consisting of x values
 * (dimensionality: rows-->number of data, cols-->number of parameter)
 */
MatrixXd X;
/**
 * y values as vector
 * (dimensionality: rows-->number of data, cols-->1)
 */
VectorXd y;

/**
* weight vector
*/
VectorXd w;

/**
* bias
*/
float bias;
};

#endif //BACHELOR_JOHANNES_SOLVER_H
