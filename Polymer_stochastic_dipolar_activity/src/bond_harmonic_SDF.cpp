/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.
  
   Implementation of stochastic dipolar extensile/contractile activity
   using Poisson process for temporal switching.
   Modified: January 20, 2026 by Ligesh
   Based on: S. Shin (PNAS 2024) active polymer framework
  
   Bond Style Syntax: bond_style harmonic/SDF
                      bond_coeff <bondtype> <k> <r0> <F_a> <tau_act> <activity_flag> <seed>
                      activity_flage => 1 Extensile & -1 Contractile

------------------------------------------------------------------------- */

#include "bond_harmonic_SDF.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "neighbor.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"
#include "update.h"

using namespace LAMMPS_NS;


BondHarmonicSDF::BondHarmonicSDF(LAMMPS *lmp) : Bond(lmp)
{
  reinitflag = 1;
  random = nullptr;
  seed = 0;
}

BondHarmonicSDF::~BondHarmonicSDF()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(spp0);
    memory->destroy(pfactor);
    memory->destroy(activity_type);
  }
  
  delete random;
}

void BondHarmonicSDF::init_style()
{

  if (!random && seed > 0) {
    random = new RanMars(lmp, seed + comm->me);
  }
}

void BondHarmonicSDF::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,rk;
  double factive; 

  ebond = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = r - r0[type];
    rk = k[type] * dr;

    if (r > 0.0) fbond = -2.0*rk/r;
    else fbond = 0.0;

    if (eflag) ebond = rk*dr;

    // activity_type = 1 for extensile, -1 for contractile
    factive = 0.0;
    if (activity_type[type] != 0 && random && pfactor[type] > 0.0) {
      if (random->uniform() < pfactor[type]) {       
        factive = activity_type[type] * spp0[type];
      }
    }

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;

      if (r > 0.0) {
        f[i1][0] -= delx/r*factive;
        f[i1][1] -= dely/r*factive;
        f[i1][2] -= delz/r*factive;
      }
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;

      if (r > 0.0) {
        f[i2][0] += delx/r*factive;
        f[i2][1] += dely/r*factive;
        f[i2][2] += delz/r*factive;
      }
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

void BondHarmonicSDF::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(r0,n+1,"bond:r0");
  memory->create(spp0,n+1,"bond:spp0");
  memory->create(pfactor,n+1,"bond:pfactor");
  memory->create(activity_type,n+1,"bond:activity_type");

  memory->create(setflag,n+1,"bond:setflag");
  for (int i = 1; i <= n; i++) {
    setflag[i] = 0;
    activity_type[i] = 0;
  }
}

void BondHarmonicSDF::coeff(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Incorrect args for bond coefficients: need 7 args (type k r0 spp0 tau_act activity_flag seed)");
  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double r0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double spp0_one = utils::numeric(FLERR,arg[3],false,lmp);
  double tau_act_one = utils::numeric(FLERR,arg[4],false,lmp);
  int activity_flag = utils::inumeric(FLERR,arg[5],false,lmp);
  int seed_one = utils::inumeric(FLERR,arg[6],false,lmp);

  if (tau_act_one <= 0.0) error->all(FLERR,"Bond tau_act must be > 0.0");
  if (activity_flag < -1 || activity_flag > 1) 
    error->all(FLERR,"Bond activity_flag must be -1 (contractile), 0 (none), or 1 (extensile)");
  if (seed_one <= 0) error->all(FLERR,"Bond random seed must be > 0");

  if (seed == 0) {
    seed = seed_one;
    delete random;
    random = new RanMars(lmp, seed + comm->me);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    r0[i] = r0_one;
    spp0[i] = spp0_one;
    pfactor[i] = 1.0 / tau_act_one; 
    activity_type[i] = activity_flag;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

double BondHarmonicSDF::equilibrium_distance(int i)
{
  return r0[i];
}

void BondHarmonicSDF::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&spp0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&pfactor[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&activity_type[1],sizeof(int),atom->nbondtypes,fp);
  fwrite(&seed,sizeof(int),1,fp);
}

void BondHarmonicSDF::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR,&k[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&r0[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&spp0[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&pfactor[1],sizeof(double),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&activity_type[1],sizeof(int),atom->nbondtypes,fp,nullptr,error);
    utils::sfread(FLERR,&seed,sizeof(int),1,fp,nullptr,error);
  }
  
  MPI_Bcast(&k[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&spp0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&pfactor[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&activity_type[1],atom->nbondtypes,MPI_INT,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);

  delete random;
  random = new RanMars(lmp, seed + comm->me);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

void BondHarmonicSDF::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g %g %g %d\n",i,k[i],r0[i],spp0[i],1.0/pfactor[i],activity_type[i]);
}


double BondHarmonicSDF::single(int type, double rsq, int /*i*/, int /*j*/,
                            double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0*rk/r;
  
  return rk*dr;
}

void *BondHarmonicSDF::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"kappa")==0) return (void*) k;
  if (strcmp(str,"r0")==0) return (void*) r0;
  if (strcmp(str,"spp0")==0) return (void*) spp0;
  if (strcmp(str,"pfactor")==0) return (void*) pfactor;
  return nullptr;
}
