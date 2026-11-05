#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include "DataUtils.h"
#include "LinearRegression.h"
#include "MemoryAnalysis.h"

static std::string dataDir = ""; // carpeta de los csv

static std::string fullPath(const std::string& name) {
    return dataDir.empty() ? name : dataDir + "/" + name;
}

// mismo flujo para simple y multiple
static void runRegression(const std::string& file, const std::string& title, bool extraQuadratic = false) {
    std::cout << "\n=========== " << title << " ===========\n";

    Matrix X;
    Vector y;
    std::vector<std::string> headers;
    if (!DataUtils::loadCSV(fullPath(file), X, y, headers)) {
        std::cout << "no pude leer " << fullPath(file) << "\n";
        return;
    }

    std::cout << "archivo: " << file << " | filas: " << X.size()
              << " | features: " << X[0].size() << "\n";

    // 80/20 con semilla 42
    Matrix Xtr, Xte;
    Vector ytr, yte;
    DataUtils::trainTestSplit(X, y, 0.2, 42, Xtr, ytr, Xte, yte);
    std::cout << "train: " << Xtr.size() << " | test: " << Xte.size() << "\n";

    std::cout << std::fixed << std::setprecision(4);

    try {
        // gradiente descendente
        LinearRegression mlr(0.1, 2000);
        mlr.fit(Xtr, ytr);

        std::cout << "\n-- gradiente descendente --\n";
        Vector w = mlr.getWeightsOriginal();
        for (size_t j = 0; j < w.size(); j++)
            std::cout << "weight[" << j << "] (" << headers[j] << "): " << w[j] << "\n";
        std::cout << "bias: " << mlr.getBiasOriginal() << "\n";

        Vector yhat = mlr.predict(Xte);
        // tabla con entradas y salidas
        std::cout << "\n  #";
        for (size_t j = 0; j < Xte[0].size(); j++) std::cout << std::setw(10) << ("x" + std::to_string(j));
        std::cout << std::setw(11) << "y_hat" << std::setw(11) << "y_real" << "\n";
        for (size_t i = 0; i < yhat.size(); i++) {
            std::cout << std::setw(3) << i + 1;
            for (size_t j = 0; j < Xte[i].size(); j++) std::cout << std::setw(10) << Xte[i][j];
            std::cout << std::setw(11) << yhat[i] << std::setw(11) << yte[i] << "\n";
        }
        std::cout << "\nscore (MSE test): " << mlr.score(Xte, yte) << "\n";
        std::cout << "RMSE test       : " << mlr.rmse(Xte, yte) << "\n";
        std::cout << "MAE test        : " << mlr.mae(Xte, yte) << "\n";
        std::cout << "R2 test         : " << mlr.r2(Xte, yte) << "\n";
        std::cout << "MSE train       : " << mlr.score(Xtr, ytr) << "\n";

        // ecuacion normal para comparar
        LinearRegression ne;
        ne.fitNormalEquation(Xtr, ytr);
        std::cout << "\n-- ecuacion normal --\n";
        Vector w2 = ne.getWeightsOriginal();
        for (size_t j = 0; j < w2.size(); j++)
            std::cout << "weight[" << j << "] (" << headers[j] << "): " << w2[j] << "\n";
        std::cout << "bias: " << ne.getBiasOriginal() << "\n";
        std::cout << "score (MSE test): " << ne.score(Xte, yte) << "\n";
        std::cout << "R2 test         : " << ne.r2(Xte, yte) << "\n";

        // extra: agrega x^2 como segunda feature
        if (extraQuadratic) {
            Matrix Xtr2 = Xtr, Xte2 = Xte;
            for (size_t i = 0; i < Xtr2.size(); i++) Xtr2[i].push_back(Xtr[i][0] * Xtr[i][0]);
            for (size_t i = 0; i < Xte2.size(); i++) Xte2[i].push_back(Xte[i][0] * Xte[i][0]);

            LinearRegression quad(0.1, 2000);
            quad.fit(Xtr2, ytr);
            std::cout << "\n-- extra: features x y x^2 --\n";
            Vector w3 = quad.getWeightsOriginal();
            std::cout << "weight[0] (x)  : " << w3[0] << "\n";
            std::cout << "weight[1] (x^2): " << w3[1] << "\n";
            std::cout << "bias: " << quad.getBiasOriginal() << "\n";
            std::cout << "score (MSE test): " << quad.score(Xte2, yte) << "\n";
            std::cout << "R2 test         : " << quad.r2(Xte2, yte) << "\n";
        }
    } catch (const std::exception& e) {
        std::cout << "error: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) dataDir = argv[1];

    int op = -1;
    while (op != 0) {
        std::cout << "\n===== Practica 2 - Regresion lineal =====\n"
                  << "1. Regresion simple (helados)\n"
                  << "2. Regresion multiple (examenes)\n"
                  << "3. Analisis de memoria\n"
                  << "4. Todo\n"
                  << "0. Salir\n"
                  << "opcion: ";
        if (!(std::cin >> op)) break;

        if (op == 1 || op == 4) runRegression("Ice_cream_selling_data.csv", "REGRESION SIMPLE", true);
        if (op == 2 || op == 4) runRegression("student_exam_scores.csv", "REGRESION MULTIPLE");
        if (op == 3 || op == 4) runMemoryAnalysis();
    }
    return 0;
}
