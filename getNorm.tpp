#include <vector>

using namespace std;

template<>
inline double getNorm(vector<vector<double>>& vv, int row, int col) { return vv[row][col]; }

template <>
inline double getNorm(vector<vector<complex<double>>>& vv, int row, int col) { return sqrt(pow(vv[row][col].real(), 2) + pow(vv[row][col].imag(), 2)); }
