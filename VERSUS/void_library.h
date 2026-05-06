int num_voids_around1(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim, 
		      int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads);

int num_voids_around2(float void_overlap, long *in_void, float R_grid2, float cell_frac, int Ncells, 
		      int xdim, int ydim, int zdim, int yzdim, int i, int j, int k);

int mark_void_region(float void_merge, long *in_void, long total_voids_found, float R_grid2, float cell_frac,
		     int Ncells, int xdim, int ydim, int zdim, int yzdim, int i, int j, int k);

int num_voids_around1_wrap(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim, 
		           int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads);

int num_voids_around2_wrap(float void_overlap, long *in_void, float R_grid2, float cell_frac, int Ncells, 
			   int xdim, int ydim, int zdim, int yzdim, int i, int j, int k);

int mark_void_region_wrap(float void_merge, long *in_void, long total_voids_found, float R_grid2, float cell_frac,
			  int Ncells, int xdim, int ydim, int zdim, int yzdim, int i, int j, int k);
