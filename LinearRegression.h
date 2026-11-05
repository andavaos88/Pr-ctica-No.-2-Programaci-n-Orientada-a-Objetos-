#pragma once
#include <vector>
#include "DataUtils.h"

class LinearRegression {
private:
    std::vector<double> weights; // un peso por feature
    double bias;                 // termino independiente
    double learningRate;         // para gradiente descendente
    int epochs;                  // iteraciones
    std::vector<double> mean_;   // medias guardadas del escalado
    std::vector<double> std_;    // desv guardadas del escalado

public:
    LinearRegression(double learningRate = 0.1, int epochs = 2000);

    // calcula media/desv de X y devuelve X escalado
    Matrix data_scaling(const Matrix& X);

    // escala con lo ya guardado (para datos de test)
    Matrix applyScaling(const Matrix& X) const;

    // entrena con gradiente descendente
    void fit(const Matrix& X, const Vector& y);

    // entrena con ecuacion normal (para comparar)
    void fitNormalEquation(const Matrix& X, const Vector& y);

    Vector predict(const Matrix& X) const;

    // score = MSE
    double score(const Matrix& X, const Vector& y) const;
    double rmse(const Matrix& X, const Vector& y) const;
    double mae(const Matrix& X, const Vector& y) const;
    double r2(const Matrix& X, const Vector& y) const;

    // pesos en la escala del modelo (datos escalados)
    const Vector& getWeights() const;
    double getBias() const;

    // pesos convertidos a las unidades originales
    Vector getWeightsOriginal() const;
    double getBiasOriginal() const;
};
