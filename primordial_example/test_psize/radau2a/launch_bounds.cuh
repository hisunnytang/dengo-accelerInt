#ifndef LAUNCH_BOUNDS_CUH
#define LAUNCH_BOUNDS_CUH
#define TARGET_BLOCK_SIZE (8)
#define TARGET_BLOCKS (8)
//shared memory active
#define SHARED_SIZE (TARGET_BLOCK_SIZE * 20 * sizeof(double))
//Large L1 cache active
#define PREFERL1
#endif
