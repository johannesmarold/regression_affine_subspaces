//
// Created by Johannes Marold on 21.05.24.
//

#include "../include/solver.h"

VectorXd solver::predict(MatrixXd Xpred) {

    return (Xpred * w).array() + bias;
}