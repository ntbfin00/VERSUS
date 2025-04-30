void mark_void_region_TEST(long *in_void, int Ncells, int dims, float R_grid2,
		      int i, int j, int k, int threads);

int num_voids_around_TEST(long total_voids_found, int dims, float middle,
		     int i, int j, int k, float *void_radius, int *void_pos,
		     float R_grid, int threads);

int num_voids_around2_TEST(int Ncells, int i, int j, int k, int dims, 
		      float R_grid2, long *in_void, int threads);
