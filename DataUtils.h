#pragma once
#include <string>
#include <vector>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

// funciones auxiliares (csv, estadistica, matrices)
namespace DataUtils {

    // lee el csv, la ultima columna es y
    bool loadCSV(const std::string& path, Matrix& X, Vector& y, std::vector<std::string>& headers);

    // mezcla con semilla fija y separa train/test
    void trainTestSplit(const Matrix& X, const Vector& y, double testRatio, unsigned int seed,
                        Matrix& Xtr, Vector& ytr, Matrix& Xte, Vector& yte);

    // estadistica
    double mean(const Vector& v);
    double stddev(const Vector& v);

    // matrices
    Matrix transpose(const Matrix& A);
    Matrix multiply(const Matrix& A, const Matrix& B);
    Vector multiplyVec(const Matrix& A, const Vector& v);

    // resuelve A x = b (gauss-jordan), false si es singular
    bool solveLinearSystem(Matrix A, Vector b, Vector& x);
}
