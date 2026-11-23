/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   Implementation of stochastic dipolar extensile/contractile activity
   using Poisson process for temporal switching.
   Modified: January 20, 2026 by Ligesh
   Based on: S. Shin (PNAS 2024) active polymer framework 

------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(harmonic/SDF,BondHarmonicSDF)

#else

#ifndef LMP_BOND_HARMONIC_SDF_H
#define LMP_BOND_HARMONIC_SDF_H

#include "bond.h"
#include "stdio.h"

namespace LAMMPS_NS {

class BondHarmonicSDF : public Bond {
 public:
  BondHarmonicSDF(class LAMMPS *);
  virtual ~BondHarmonicSDF();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  virtual void read_restart(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  virtual void *extract(const char *, int &);
  void init_style();

 protected:
  double *k, *r0;
  double *spp0;       
  double *pfactor;   
  int *activity_type; 
  
  class RanMars *random;
  int seed;
  
  virtual void allocate();
};

}

#endif
#endif
