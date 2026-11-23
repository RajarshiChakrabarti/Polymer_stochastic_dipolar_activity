#ifdef BOND_CLASS
BondStyle(harmonic/SDF,BondHarmonicSDF)
#else

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
#include "utils.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondHarmonicSDF::BondHarmonicSDF(LAMMPS *lmp) : Bond(lmp)
{
  random = nullptr;
  seed = 0;

  bond_state = nullptr;
  bond_timer = nullptr;
}

/* ---------------------------------------------------------------------- */

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

  if (bond_state) memory->destroy(bond_state);
  if (bond_timer) memory->destroy(bond_timer);

  delete random;
}

/* ---------------------------------------------------------------------- */

void BondHarmonicSDF::init_style()
{
  if (!random && seed > 0)
    random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

void BondHarmonicSDF::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,fbond,ebond;
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

  // SAFE memory (per atom)
  if (bond_state == nullptr) {
    int nmax = atom->nmax;
    memory->create(bond_state, nmax, "bond_state");
    memory->create(bond_timer, nmax, "bond_timer");

    for (int i = 0; i < nmax; i++) {
      bond_state[i] = 0;
      bond_timer[i] = 0;
    }
  }

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

    fbond = (r > 0.0) ? -2.0 * rk / r : 0.0;
    if (eflag) ebond = rk * dr;

    factive = 0.0;

    if (activity_type[type] != 0 && random && pfactor[type] > 0.0) {

      if (bond_state[i1] == 0) {
        if (random->uniform() < pfactor[type]) {
          bond_state[i1] = 1;
          bond_timer[i1] = tau_p_steps;
        }
      }

      if (bond_state[i1] == 1) {
        factive = activity_type[type] * spp0[type];

        bond_timer[i1]--;
        if (bond_timer[i1] <= 0)
          bond_state[i1] = 0;
      }
    }

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;

      if (r > 0.0) {
        f[i1][0] -= delx/r * factive;
        f[i1][1] -= dely/r * factive;
        f[i1][2] -= delz/r * factive;
      }
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;

      if (r > 0.0) {
        f[i2][0] += delx/r * factive;
        f[i2][1] += dely/r * factive;
        f[i2][2] += delz/r * factive;
      }
    }

    if (evflag)
      ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

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
}

/* ---------------------------------------------------------------------- */

void BondHarmonicSDF::coeff(int narg, char **arg)
{
  if (narg != 8)
    error->all(FLERR,"bond_coeff needs 8 args");

  if (!allocated) allocate();

  int ilo,ihi;
  utils::bounds(FLERR,arg[0],1,atom->nbondtypes,ilo,ihi,error);

  double k_one = utils::numeric(FLERR,arg[1],false,lmp);
  double r0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double spp0_one = utils::numeric(FLERR,arg[3],false,lmp);
  double tau_act_one = utils::numeric(FLERR,arg[4],false,lmp);
  int activity_flag = utils::inumeric(FLERR,arg[5],false,lmp);
  int seed_one = utils::inumeric(FLERR,arg[6],false,lmp);
  double tau_p = utils::numeric(FLERR,arg[7],false,lmp);

  tau_p_steps = tau_p / update->dt;

  seed = seed_one;
  delete random;
  random = new RanMars(lmp, seed + comm->me);

  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    r0[i] = r0_one;
    spp0[i] = spp0_one;
    pfactor[i] = 1.0 / tau_act_one;
    activity_type[i] = activity_flag;
    setflag[i] = 1;
  }
}

/* ---------------------------------------------------------------------- */

double BondHarmonicSDF::equilibrium_distance(int i)
{
  return r0[i];
}

/* ---------------------------------------------------------------------- */

double BondHarmonicSDF::single(int type, double rsq, int i, int j, double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;

  fforce = (r > 0.0) ? -2.0 * rk / r : 0.0;
  return rk * dr;
}

/* ---------------------------------------------------------------------- */

// MINIMAL SAFE restart (do nothing)

void BondHarmonicSDF::write_restart(FILE *fp) {}

void BondHarmonicSDF::read_restart(FILE *fp) {}

#endif
