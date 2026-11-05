#include "DataUtils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

namespace DataUtils {

    // parte una linea por comas
    static std::vector<std::string> splitLine(const std::string& line) {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) out.push_back(cell);
        return out;
    }

    // quita \r del final
    static void removeCR(std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    }

    // deja solo ascii (el simbolo de grados daba problemas)
    static std::string onlyAscii(const std::string& s) {
        std::string r;
        for (size_t i = 0; i < s.size(); i++)
            if (static_cast<unsigned char>(s[i]) < 128) r += s[i];
        return r;
    }

    bool loadCSV(const std::string& path, Matrix& X, Vector& y, std::vector<std::string>& headers) {
        std::ifstream file(path.c_str());
        if (!file.is_open()) return false;

        X.clear(); y.clear(); headers.clear();

        std::string line;
        if (!std::getline(file, line)) return false;
        removeCR(line);
        std::vector<std::string> raw = splitLine(line);
        for (size_t i = 0; i < raw.size(); i++) headers.push_back(onlyAscii(raw[i]));

        while (std::getline(file, line)) {
            removeCR(line);
            if (line.empty()) continue;

            std::vector<std::string> cells = splitLine(line);
            if (cells.size() != headers.size()) return false; // fila rara

            Vector row;
            try {
                for (size_t j = 0; j + 1 < cells.size(); j++) row.push_back(std::stod(cells[j]));
                y.push_back(std::stod(cells.back()));
            } catch (...) {
                return false; // no era numero
            }
            X.push_back(row);
        }
        return !X.empty();
    }

    void trainTestSplit(const Matrix& X, const Vector& y, double testRatio, unsigned int seed,
                        Matrix& Xtr, Vector& ytr, Matrix& Xte, Vector& yte) {
        size_t n = X.size();
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; i++) idx[i] = i;

        // fisher-yates a mano, asi sale igual en cualquier compilador
        std::mt19937 rng(seed);
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = rng() % (i + 1);
            std::swap(idx[i], idx[j]);
        }

        size_t nTest = static_cast<size_t>(n * testRatio + 0.5);

        Xtr.clear(); ytr.clear(); Xte.clear(); yte.clear();
        for (size_t k = 0; k < n; k++) {
            if (k < nTest) { Xte.push_back(X[idx[k]]); yte.push_back(y[idx[k]]); }
            else           { Xtr.push_back(X[idx[k]]); ytr.push_back(y[idx[k]]); }
        }
    }

    double mean(const Vector& v) {
        double s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += v[i];
        return v.empty() ? 0.0 : s / v.size();
    }

    // desviacion poblacional
    double stddev(const Vector& v) {
        double m = mean(v), s = 0.0;
        for (size_t i = 0; i < v.size(); i++) s += (v[i] - m) * (v[i] - m);
        return v.empty() ? 0.0 : std::sqrt(s / v.size());
    }

    Matrix transpose(const Matrix& A) {
        if (A.empty()) return Matrix();
        Matrix T(A[0].size(), Vector(A.size(), 0.0));
        for (size_t i = 0; i < A.size(); i++)
            for (size_t j = 0; j < A[0].size(); j++)
                T[j][i] = A[i][j];
        return T;
    }

    Matrix multiply(const Matrix& A, const Matrix& B) {
        Matrix C(A.size(), Vector(B[0].size(), 0.0));
        for (size_t i = 0; i < A.size(); i++)
            for (size_t k = 0; k < B.size(); k++)
                for (size_t j = 0; j < B[0].size(); j++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    Vector multiplyVec(const Matrix& A, const Vector& v) {
        Vector r(A.size(), 0.0);
        for (size_t i = 0; i < A.size(); i++)
            for (size_t j = 0; j < v.size(); j++)
                r[i] += A[i][j] * v[j];
        return r;
    }

    bool solveLinearSystem(Matrix A, Vector b, Vector& x) {
        size_t n = A.size();
        for (size_t i = 0; i < n; i++) A[i].push_back(b[i]); // matriz aumentada

        for (size_t col = 0; col < n; col++) {
            // pivote parcial
            size_t piv = col;
            for (size_t r = col + 1; r < n; r++)
                if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
            if (std::fabs(A[piv][col]) < 1e-12) return false;
            std::swap(A[col], A[piv]);

            // fila pivote a 1
            double d = A[col][col];
            for (size_t j = col; j <= n; j++) A[col][j] /= d;

            // ceros en las demas filas
            for (size_t r = 0; r < n; r++) {
                if (r == col) continue;
                double f = A[r][col];
                for (size_t j = col; j <= n; j++) A[r][j] -= f * A[col][j];
            }
        }

        x.assign(n, 0.0);
        for (size_t i = 0; i < n; i++) x[i] = A[i][n];
        return true;
    }
}
