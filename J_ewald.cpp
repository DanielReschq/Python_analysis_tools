#include <vector>
#include <math.h>

extern "C" double ewald_summation_1D(size_t L, int dx, double alpha, int n_cutoff)
{
    double J_ewald = 0.;
    long long r = 0;

    int sgnx = (dx > 0) ? 1 : (dx < 0) ? -1
                                       : 0;
    
    if (size_t(abs(2 * dx)) > L)
    {
        dx = -sgnx * (L - abs(dx));
    }

    for (int n = -n_cutoff; n < n_cutoff + 1; n++)
    {
        r = dx + (long long)L * n;
        J_ewald += pow(std::abs(r), -1. * (alpha));
    }

    return J_ewald;
}

extern "C" double ewald_summation_2D(size_t Lx, size_t Ly, int dx, int dy, double alpha, int n_cutoff)
{
    double J_ewald = 0., r = 0.;

    int sgnx = (dx > 0) ? 1 : (dx < 0) ? -1
                                       : 0;
    int sgny = dy > 0 ? 1 : dy < 0 ? -1
                                   : 0;

    if (size_t(abs(2 * dx)) > Lx)
    {
        dx = -sgnx * (Lx - abs(dx));
    }
    if (size_t(abs(2 * dy)) > Ly)
    {
        dy = -sgny * (Ly - abs(dy));
    }

    std::vector<long long> vec = {0, 0};
    for (int m = -n_cutoff; m < n_cutoff + 1; m++)
    {
        vec[0] = dx + Lx * (long long)m;
        for (int n = -n_cutoff; n < n_cutoff + 1; n++)
        {
            vec[1] = dy + Ly * (long long)n;

            r = sqrt((double)(vec[0] * vec[0] + vec[1] * vec[1]));
            J_ewald += pow(r, -1. * (alpha));
        }
    }

    return J_ewald;
}