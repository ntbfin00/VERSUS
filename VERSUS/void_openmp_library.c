#include <stdio.h>
#include "void_openmp_library.h"
#include <omp.h>
#include <math.h>


// Determine the volume overlap fraction between two spheres
// Rv - Radius of previously detected void, Rc - Radius of void candidate, d2 - Squared distance between void and candidate centres.
float calc_void_overlap(float Rv, float Rc, float d2)
{
  float a, b, c, d;

  // no overlap if voids are not touching
  if (d2 >= ((Rv + Rc)*(Rv + Rc)))  return 0.;
  
  d = sqrt(d2);
  // expression from: https://mathworld.wolfram.com/Sphere-SphereIntersection.html
  a = (Rv + Rc - d) * (Rv + Rc - d);
  b = (d*d + 2*d*Rc - 3*Rc*Rc + 2*d*Rv + 6*Rc*Rv - 3*Rv*Rv);
  c = 1 / (16 * d * Rc*Rc*Rc);
  
  return a * b * c;
}


// This routine computes the distance between a cell and voids already identified
// if that distance is smaller than the sum of their radii then the cell can not
// host a void as it will overlap with the other void
int num_voids_around1_wrap(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim,
		      	   int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads)
{

  int l, nearby_voids=0;
  int dx, dy, dz, dist2;
  float overlap_frac, tot_overlap=0.;

#pragma omp parallel for num_threads(threads) private(l,dx,dy,dz,dist2)
  for (l=0; l<total_voids_found; l++)
    {
      // exit if void volume overlap greater than threshold 
      if (tot_overlap>void_overlap)
	{
	 nearby_voids += 1;
	 continue;
	}

      dx = i - void_pos[3*l+0];
      if (dx>xdim/2)   dx = dx - xdim;
      if (dx<-xdim/2)  dx = dx + xdim;

      dy = j - void_pos[3*l+1];
      if (dy>ydim/2)   dy = dy - ydim;
      if (dy<-ydim/2)  dy = dy + ydim;

      dz = k - void_pos[3*l+2];
      if (dz>zdim/2)   dz = dz - zdim;
      if (dz<-zdim/2)  dz = dz + zdim;

      dist2 = dx*dx + dy*dy + dz*dz;

      if (dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid)))
	{
	overlap_frac = calc_void_overlap(void_radius[l], R_grid, dist2);
#pragma omp atomic
	  tot_overlap += overlap_frac;
	  // nearby_voids += 1;
	}
    }
  
  return nearby_voids;
}


// This routine computes the distance between a cell and voids already identified
// if that distance is smaller than the sum of their radii then the cell can not
// host a void as it will overlap with the other void
int num_voids_around1(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim,
		      int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads)
{

  int l, nearby_voids=0;
  int dx, dy, dz, dist2;
  float overlap_frac, tot_overlap=0.;

#pragma omp parallel for num_threads(threads) private(l,dx,dy,dz,dist2)
  for (l=0; l<total_voids_found; l++)
    {
      // exit if void volume overlap greater than threshold 
      if (tot_overlap>void_overlap)
	{
	 nearby_voids += 1;
	 continue;
	}

      dx = i - void_pos[3*l+0];
      dy = j - void_pos[3*l+1];
      dz = k - void_pos[3*l+2];

      dist2 = dx*dx + dy*dy + dz*dz;

      if (dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid)))
	{
	overlap_frac = calc_void_overlap(void_radius[l], R_grid, dist2);
#pragma omp atomic
	  tot_overlap += overlap_frac;
	  // nearby_voids += 1;
	}
    }
  
  return nearby_voids;
}


// This routine looks at the cells around a given cell to see if those belong
// to other voids
int num_voids_around2_wrap(float void_overlap, int Ncells, int i, int j, int k, int xdim, int ydim, int zdim, int yzdim,
		      	   float R_grid, float R_grid2, char *in_void, int threads)
{
  int l, m, n, i1, j1, k1, nearby_voids=0;
  long num;
  float dist2, overlap_frac, tot_overlap=0.;

#pragma omp parallel for num_threads(threads) private(l, m, n, i1, j1, k1, num, dist2)
  for (l=-Ncells; l<=Ncells; l++)
    {
      // exit if void volume overlap greater than threshold 
      if (tot_overlap>void_overlap)
	{
	 nearby_voids += 1;
	 continue;
	}

      //i1 = (i+l+dims)%dims;
      i1 = i+l;
      if (i1>=xdim) i1 = i1-xdim;
      if (i1<0)     i1 = i1+xdim;

      for (m=-Ncells; m<=Ncells; m++)
	{

	  //j1 = (j+m+dims)%dims;
	  j1 = j+m;
	  if (j1>=ydim) j1 = j1-ydim;
	  if (j1<0)     j1 = j1+ydim;

	  for (n=-Ncells; n<=Ncells; n++)
	    {

	      //k1 = (k+n+dims)%dims;
	      k1 = k+n;
	      if (k1>=zdim) k1 = k1-zdim;
	      if (k1<0)     k1 = k1+zdim;

	      num = yzdim*i1 + zdim*j1 + k1;

	      if (in_void[num]==0)  continue;
	      else 
		{
		  dist2 = l*l + m*m + n*n;
		  if (dist2<R_grid2)
		    {
		      overlap_frac = 3 / (4 * M_PI * R_grid*R_grid*R_grid);
#pragma omp atomic
		      tot_overlap += overlap_frac;
		      // nearby_voids += 1;
		    }
		}
	    }
	}
    }
	  
  return nearby_voids;
}


// This routine looks at the cells around a given cell to see if those belong
// to other voids
int num_voids_around2(float void_overlap, int Ncells, int i, int j, int k, int xdim, int ydim, int zdim, int yzdim,
		      	   float R_grid, float R_grid2, char *in_void, int threads)
{
  int l, m, n, i1, j1, k1, nearby_voids=0;
  long num;
  float dist2, overlap_frac, tot_overlap=0.;

#pragma omp parallel for num_threads(threads) private(l, m, n, i1, j1, k1, num, dist2)
  for (l=-Ncells; l<=Ncells; l++)
    {
      // exit if void volume overlap greater than threshold 
      if (tot_overlap>void_overlap)
	{
	 nearby_voids += 1;
	 continue;
	}
      i1 = i+l;
      for (m=-Ncells; m<=Ncells; m++)
	{
	  j1 = j+m;
	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      k1 = k+n;

	      num = yzdim*i1 + zdim*j1 + k1;

	      if (in_void[num]==0)  continue;
	      else 
		{
		  dist2 = l*l + m*m + n*n;
		  if (dist2<R_grid2)
		    {
		      overlap_frac = 3 / (4 * M_PI * R_grid*R_grid*R_grid);
#pragma omp atomic
		      tot_overlap += overlap_frac;
		      // nearby_voids += 1;
		    }
		}
	    }
	}
    }
	  
  return nearby_voids;
}


void mark_void_region_wrap(char *in_void, int Ncells, int xdim, int ydim, int zdim, 
			   int yzdim, float R_grid2, int i, int j, int k, int threads)
{
  int l, m, n, i1, j1, k1;
  long number;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l,m,n,i1,j1,k1,dist2,number) firstprivate(i,j,k,Ncells,R_grid2,xdim,ydim,zdim,yzdim) shared(in_void)
  for (l=-Ncells; l<=Ncells; l++)
    {
      //i1 = (i+l+dims)%dims;
      i1 = i+l;
      if (i1>=xdim) i1 = i1-xdim;
      if (i1<0)     i1 = i1+xdim;
		      
      for (m=-Ncells; m<=Ncells; m++)
	{
	  //j1 = (j+m+dims)%dims;
	  j1 = j+m;
	  if (j1>=ydim) j1 = j1-ydim;
	  if (j1<0)     j1 = j1+ydim;

	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      //k1 = (k+n+dims)%dims;
	      k1 = k+n;
	      if (k1>=zdim) k1 = k1-zdim;
	      if (k1<0)     k1 = k1+zdim;

	      dist2 = l*l + m*m + n*n;
	      if (dist2<R_grid2)
		{
		  number = yzdim*i1 + zdim*j1 + k1;
		  in_void[number] = 1;
		}
	    }
	}
    } 
}


void mark_void_region(char *in_void, int Ncells, int xdim, int ydim, int zdim, 
		      int yzdim, float R_grid2, int i, int j, int k, int threads)
{
  int l, m, n, i1, j1, k1;
  long number;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l,m,n,i1,j1,k1,dist2,number) firstprivate(i,j,k,Ncells,R_grid2,xdim,ydim,zdim,yzdim) shared(in_void)
  for (l=-Ncells; l<=Ncells; l++)
    {
      i1 = i+l;
		      
      for (m=-Ncells; m<=Ncells; m++)
	{
	  j1 = j+m;

	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      k1 = k+n;

	      dist2 = l*l + m*m + n*n;
	      if (dist2<R_grid2)
		{
		  number = yzdim*i1 + zdim*j1 + k1;
		  in_void[number] = 1;
		}
	    }
	}
    } 
}

