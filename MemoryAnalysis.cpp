#include "MemoryAnalysis.h"
#include "LinearRegression.h"
#include <iomanip>
#include <iostream>
#include <vector>

// objeto global (vive en la zona de datos)
LinearRegression globalModel;
int globalInt = 7; // para comparar con el objeto

// entrena con un dataset chiquito para que los pesos ocupen memoria
static void tinyFit(LinearRegression& m) {
    Matrix X;
    Vector y;
    X.push_back(Vector{1, 2});
    X.push_back(Vector{2, 1});
    X.push_back(Vector{3, 5});
    X.push_back(Vector{4, 3});
    X.push_back(Vector{5, 8});
    y.push_back(5); y.push_back(4); y.push_back(14); y.push_back(11); y.push_back(21);
    m.fit(X, y);
}

static void showObject(const char* label, const LinearRegression& m) {
    std::cout << "  " << label << "\n";
    std::cout << "    direccion del objeto : " << static_cast<const void*>(&m) << "\n";
    std::cout << "    sizeof(objeto)       : " << sizeof(m) << " bytes\n";
    std::cout << "    buffer de weights    : " << static_cast<const void*>(m.getWeights().data())
              << " (" << m.getWeights().size() * sizeof(double) << " bytes en el heap)\n";
}

void runMemoryAnalysis() {
    std::cout << "\n=========== ANALISIS DE MEMORIA ===========\n";

    // info del entorno
    std::cout << "\n[0] Entorno\n";
#if defined(_MSC_VER)
    std::cout << "  compilador : MSVC (" << _MSC_VER << ")\n";
#elif defined(__clang__)
    std::cout << "  compilador : clang\n";
#elif defined(__GNUC__)
    std::cout << "  compilador : GCC " << __VERSION__ << "\n";
#endif
#ifdef NDEBUG
    std::cout << "  modo       : Release\n";
#else
    std::cout << "  modo       : Debug\n";
#endif
    std::cout << "  sizeof(void*) = " << sizeof(void*) << " (" << sizeof(void*) * 8 << " bits)\n";

    // tamanos de los miembros
    std::cout << "\n[1] Tamanos de los miembros\n";
    std::cout << "  sizeof(std::vector<double>) = " << sizeof(std::vector<double>) << "\n";
    std::cout << "  sizeof(double)              = " << sizeof(double) << "\n";
    std::cout << "  sizeof(int)                 = " << sizeof(int) << "\n";
    std::cout << "  sizeof(LinearRegression)    = " << sizeof(LinearRegression) << "\n";
    std::cout << "  miembros: 3 vectores + 2 double + 1 int = "
              << 3 * sizeof(std::vector<double>) + 2 * sizeof(double) + sizeof(int)
              << " (el resto es padding)\n";

    // memoria global, local y dinamica
    std::cout << "\n[2] Global, local y dinamica\n";

    tinyFit(globalModel);
    showObject("GLOBAL", globalModel);

    LinearRegression localModel; // en el stack
    tinyFit(localModel);
    showObject("LOCAL (stack)", localModel);

    LinearRegression* dynModel = new LinearRegression(); // en el heap
    tinyFit(*dynModel);
    showObject("DINAMICO (heap)", *dynModel);
    std::cout << "    el puntero dynModel  : " << static_cast<const void*>(&dynModel)
              << " (vive en el stack, apunta a " << static_cast<const void*>(dynModel) << ")\n";

    // referencias para ubicar las zonas
    std::cout << "\n  referencias de zonas\n";
    std::cout << "    codigo (runMemoryAnalysis) : "
              << reinterpret_cast<const void*>(&runMemoryAnalysis) << "\n";
    std::cout << "    global int                 : " << static_cast<const void*>(&globalInt) << "\n";
    int localInt = 1;
    std::cout << "    local int                  : " << static_cast<const void*>(&localInt) << "\n";
    int* heapInt = new int(1);
    std::cout << "    heap int                   : " << static_cast<const void*>(heapInt) << "\n";

    delete dynModel; // liberar
    dynModel = nullptr;

    // vector de LinearRegression en memoria dinamica
    std::cout << "\n[3] Vector de LinearRegression en el heap\n";
    std::vector<LinearRegression>* vec = new std::vector<LinearRegression>(5);
    for (size_t i = 0; i < vec->size(); i++) tinyFit((*vec)[i]);

    std::cout << "  direccion del objeto vector : " << static_cast<const void*>(vec) << "\n";
    std::cout << "  sizeof(*vec)                : " << sizeof(*vec) << " bytes (solo el 'encabezado')\n";
    std::cout << "  size() / capacity()         : " << vec->size() << " / " << vec->capacity() << "\n";
    std::cout << "  buffer de elementos (data)  : " << static_cast<const void*>(vec->data()) << "\n";
    std::cout << "  bytes del buffer            : " << vec->size() << " x " << sizeof(LinearRegression)
              << " = " << vec->size() * sizeof(LinearRegression) << "\n";
    for (size_t i = 0; i < vec->size(); i++) {
        std::cout << "    elemento " << i << " en " << static_cast<const void*>(&(*vec)[i]);
        if (i > 0)
            std::cout << "  (+" << reinterpret_cast<const char*>(&(*vec)[i]) -
                                   reinterpret_cast<const char*>(&(*vec)[i - 1]) << " bytes)";
        std::cout << "\n";
    }
    delete vec;
    vec = nullptr;
    delete heapInt; // este se libera al final
    heapInt = nullptr;

    // arreglo con new[]
    std::cout << "\n[4] Arreglo con new[]\n";
    LinearRegression* arr = new LinearRegression[3];
    std::cout << "  direccion del arreglo : " << static_cast<const void*>(arr) << "\n";
    std::cout << "  sizeof(arr) (puntero) : " << sizeof(arr) << " bytes\n";
    std::cout << "  bytes reales          : 3 x " << sizeof(LinearRegression) << " = "
              << 3 * sizeof(LinearRegression) << "\n";
    delete[] arr;
    arr = nullptr;

    std::cout << "\n(memoria liberada)\n";
}
