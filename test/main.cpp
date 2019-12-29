
#include "NLSSolver.hpp"

#include <random>

using namespace std;

int main(int argc, char** argv)
{
    //True states
    double a = 0.1, b = 0.5, c = 2.0;
    int N = 100;
    double w_sigma = 0.1;
    default_random_engine generator;
    normal_distribution<double> noise(0.,w_sigma);

    nlssolver::NLSSolver solver(0.0,0.0,0.0);
    solver.setEstimationPrecision(1e-6);
    solver.setMaximumIterations(30);

    //generate random observations
    for(int i = 0; i < N; ++i){
        double x = i / 100.0;
        double n = noise(generator);
        double y = exp(a*x*x+b*x+c)+n;
        solver.addObservation(x,y);
    }

    //solve by gauss-newton
    solver.solveByGN();

    //solve by LM
    solver.setInitialStates(0.,0.,0.);
    solver.solveByLM();

    //solve by dogleg
    solver.setInitialStates(0.,0.,0.);
    solver.solveByDogLeg();

    return 0;

}