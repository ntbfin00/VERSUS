int num_voids_around1(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim, 
		      int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads);

int num_voids_around2(float void_overlap, int Ncells, int i, int j, int k, int xdim, int ydim, int zdim, 
		      int yzdim, float R_grid, float R_grid2, long *in_void, int threads);

void mark_void_region(long *in_void, int Ncells, int xdim, int ydim, int zdim, 
		      int yzdim, float R_grid2, int i, int j, int k, int threads);

int num_voids_around1_wrap(float void_overlap, long total_voids_found, int xdim, int ydim, int zdim, 
		           int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int threads);

int num_voids_around2_wrap(float void_overlap, int Ncells, int i, int j, int k, int xdim, int ydim, int zdim, 
		           int yzdim, float R_grid, float R_grid2, long *in_void, int threads);

void mark_void_region_wrap(long *in_void, int Ncells, int xdim, int ydim, int zdim, 
		           int yzdim, float R_grid2, int i, int j, int k, int threads);
