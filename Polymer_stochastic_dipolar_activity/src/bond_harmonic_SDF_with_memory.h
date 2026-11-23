#ifdef BOND_CLASS

BondStyle(harmonic/SDF,BondHarmonicSDF)

#else

#ifndef LMP_BOND_HARMONIC_SDF_H
#define LMP_BOND_HARMONIC_SDF_H

#include "bond.h"

namespace LAMMPS_NS {

class BondHarmonicSDF : public Bond {
 public:
  BondHarmonicSDF(class LAMMPS *);
  virtual ~BondHarmonicSDF();

  void compute(int, int);
  void coeff(int, char **);
  void init_style();

  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  double single(int, double, int, int, double &);

 protected:
  double *k, *r0;
  double *spp0;
  double *pfactor;
  int *activity_type;

  int *bond_state;
  int *bond_timer;
  int maxbonds;
  int tau_p_steps;

  class RanMars *random;
  int seed;

  void allocate();
};

}

#endif
#endif
