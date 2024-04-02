#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace Eigen;
using namespace std;

double lusol(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xLU)
{
    xLU=A.partialPivLu().solve(b);
    double erroreLU = (sol-xLU).norm()/sol.norm();
    return erroreLU;
}

double qrsol(const MatrixXd& A, const VectorXd& b, const VectorXd& sol, Vector2d& xQR)
{
    xQR=A.colPivHouseholderQr().solve(b);
    double erroreQR=(sol-xQR).norm()/sol.norm();
    return erroreQR;
}

int main() {
        Matrix2d A;
    Vector2d b;
    Vector2d xLU;
    Vector2d xQR;

    Vector2d sol;
    sol << -1.0e+00,-1.0e+00;
    // System 1
    A << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b << -5.169911863249772e-01, 1.672384680188350e-01;

    Vector2d x1LU;
    double err1 = lusol(A,b,sol,x1LU);
    cout <<scientific << setprecision(2)<< "primo sistema:\n"<<"xLU=[ " << x1LU(0)<< ";"<<x1LU(1) << "]\n" << "errore relativo: " << err1 << endl;

    Vector2d x1QR;
    double err1QR= qrsol(A,b,sol,x1QR);
    cout << scientific << setprecision(2)<<"xQR=[ " << x1QR(0)<<";"<<x1QR(1)<< "]\n" <<"errore relativo: " << err1QR << endl;

    // System 2
    A << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b << -6.394645785530173e-04, 4.259549612877223e-04;

    Vector2d x2LU;
    double err2 = lusol(A,b,sol,x2LU);
    cout <<scientific << setprecision(2)<< "secondo sistema:\n"<<"xLU=[ " << x2LU(0)<<";"<<x2LU(1) << "]\n" << "errore relativo: " << err2 << endl;

    Vector2d x2QR;
    double err2QR= qrsol(A,b,sol,x2QR);
    cout << scientific << setprecision(2)<<"xQR=[ " << x2QR(0)<<";" <<x2QR(1) << "]\n" <<"errore relativo: " << err2QR << endl;

    // System 3
    A << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b << -6.400391328043042e-10, 4.266924591433963e-10;
    // Solve system 3 using PALU decomposition
    Vector2d x3LU;
    double err3 = lusol(A,b,sol,x3LU);
    cout <<scientific << setprecision(2)<< "terzo sistema:\n"<< "xLU=[ " << x3LU(0)<<";"<<x3LU(1) << "]\n" << "errore relativo: " << err3 << endl;

    Vector2d x3QR;
    double err3QR= qrsol(A,b,sol,x3QR);
    cout << scientific << setprecision(2)<<"xQR=[ " << x3QR(0)<<";"<<x3QR(1) << "]\n" <<"errore relativo: " << err3QR << endl;

    return 0;
}
