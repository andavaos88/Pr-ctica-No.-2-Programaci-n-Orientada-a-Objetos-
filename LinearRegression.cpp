#include "LinearRegression.h"
#include <cmath>
#include <stdexcept>

LinearRegression::LinearRegression(double learningRate, int epochs)
    : bias(0.0), learningRate(learningRate), epochs(epochs) {
    // vectores vacios hasta que se entrene
}

Matrix LinearRegression::data_scaling(const Matrix& X) {
    if (X.empty()) throw std::invalid_argument("X vacia");
    size_t n = X.size(), p = X[0].size();

    mean_.assign(p, 0.0);
    std_.assign(p, 1.0);

    for (size_t j = 0; j < p; j++) {
        Vector col(n);
        for (size_t i = 0; i < n; i++) col[i] = X[i][j];
        mean_[j] = DataUtils::mean(col);
        double s = DataUtils::stddev(col);
        std_[j] = (s < 1e-12) ? 1.0 : s; // evita dividir por 0
    }
    return applyScaling(X);
}

Matrix LinearRegression::applyScaling(const Matrix& X) const {
    Matrix Xs = X;
    for (size_t i = 0; i < X.size(); i++) {
        if (X[i].size() != mean_.size()) throw std::invalid_argument("columnas no coinciden");
        for (size_t j = 0; j < X[i].size(); j++)
            Xs[i][j] = (X[i][j] - mean_[j]) / std_[j];
    }
    return Xs;
}

void LinearRegression::fit(const Matrix& X, const Vector& y) {
    if (X.size() != y.size()) throw std::invalid_argument("X y y distinto tamano");

    Matrix Xs = data_scaling(X);
    size_t n = Xs.size(), p = Xs[0].size();

    weights.assign(p, 0.0);
    bias = 0.0;

    for (int e = 0; e < epochs; e++) {
        Vector gw(p, 0.0);
        double gb = 0.0;

        // gradiente del mse
        for (size_t i = 0; i < n; i++) {
            double pred = bias;
            for (size_t j = 0; j < p; j++) pred += weights[j] * Xs[i][j];
            double err = pred - y[i];
            gb += err;
            for (size_t j = 0; j < p; j++) gw[j] += err * Xs[i][j];
        }

        // actualiza
        bias -= learningRate * (2.0 / n) * gb;
        for (size_t j = 0; j < p; j++) weights[j] -= learningRate * (2.0 / n) * gw[j];
    }
}

void LinearRegression::fitNormalEquation(const Matrix& X, const Vector& y) {
    if (X.size() != y.size()) throw std::invalid_argument("X y y distinto tamano");

    Matrix Xs = data_scaling(X);
    size_t n = Xs.size(), p = Xs[0].size();

    // columna de 1s para el bias
    Matrix A(n, Vector(p + 1, 1.0));
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < p; j++) A[i][j + 1] = Xs[i][j];

    Matrix At = DataUtils::transpose(A);
    Matrix AtA = DataUtils::multiply(At, A);
    Vector Aty = DataUtils::multiplyVec(At, y);

    Vector sol;
    if (!DataUtils::solveLinearSystem(AtA, Aty, sol))
        throw std::runtime_error("matriz singular");

    bias = sol[0];
    weights.assign(sol.begin() + 1, sol.end());
}

Vector LinearRegression::predict(const Matrix& X) const {
    if (weights.empty()) throw std::runtime_error("modelo sin entrenar");

    Matrix Xs = applyScaling(X);
    Vector yhat(Xs.size(), 0.0);
    for (size_t i = 0; i < Xs.size(); i++) {
        double s = bias;
        for (size_t j = 0; j < weights.size(); j++) s += weights[j] * Xs[i][j];
        yhat[i] = s;
    }
    return yhat;
}

double LinearRegression::score(const Matrix& X, const Vector& y) const {
    Vector yhat = predict(X);
    double s = 0.0;
    for (size_t i = 0; i < y.size(); i++) s += (yhat[i] - y[i]) * (yhat[i] - y[i]);
    return s / y.size();
}

double LinearRegression::rmse(const Matrix& X, const Vector& y) const {
    return std::sqrt(score(X, y));
}

double LinearRegression::mae(const Matrix& X, const Vector& y) const {
    Vector yhat = predict(X);
    double s = 0.0;
    for (size_t i = 0; i < y.size(); i++) s += std::fabs(yhat[i] - y[i]);
    return s / y.size();
}

double LinearRegression::r2(const Matrix& X, const Vector& y) const {
    Vector yhat = predict(X);
    double ym = DataUtils::mean(y), ssRes = 0.0, ssTot = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        ssRes += (y[i] - yhat[i]) * (y[i] - yhat[i]);
        ssTot += (y[i] - ym) * (y[i] - ym);
    }
    return 1.0 - ssRes / ssTot;
}

const Vector& LinearRegression::getWeights() const { return weights; }
double LinearRegression::getBias() const { return bias; }

Vector LinearRegression::getWeightsOriginal() const {
    Vector w(weights.size());
    for (size_t j = 0; j < weights.size(); j++) w[j] = weights[j] / std_[j];
    return w;
}

double LinearRegression::getBiasOriginal() const {
    double b = bias;
    for (size_t j = 0; j < weights.size(); j++) b -= weights[j] * mean_[j] / std_[j];
    return b;
}
