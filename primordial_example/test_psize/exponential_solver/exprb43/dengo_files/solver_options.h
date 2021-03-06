
        /*! ile

        rief A file generated by Scons that specifies various options to the solvers

        This file, autogenerated by SCons contains a number of definitions that can be
        enabled via command line options.  These control the integrator behaviour and
        alternately enable special behaviour such as logging, divergence measuring, etc.

        Note that it is not typical for all these options to be turned on at the same time,
        but it is done here for documentation purposes.

        */
        #ifndef SOLV_OPT_HEAD
        #define SOLV_OPT_HEAD

        /* Tolerances and Timestep */

        #include <float.h>

        /*! Absolute solver tolerance */
        //#define ATOL (1e-10)
        /*! Relative solver tolerance */
        //#define RTOL (1e-6)
        /*! Solver timestep (may be used to force multiple timesteps per global integration step) */
        //#define t_step (1e-6)
        /*! Global integration timestep */
        //#define end_time (1e-3)

        /*! Machine precision constant. */
        #define EPS DBL_EPSILON
        /*! Smallest representable double */
        #define SMALL DBL_MIN

        /** type of rational approximant (n, n) */
        #define N_RA (10)

        
        /* CVodes Parameters */
        //#define CV_MAX_ORD (5) //maximum order for method, default for BDF is 5
        /*! Maximum steps the solver will take in one timestep set to -1 (disabled) by default */
        #define CV_MAX_STEPS (20000)
        //#define CV_HMAX (0)  //upper bound on step size (integrator step, not global timestep)
        //#define CV_HMIN (0) //lower bound on step size (integrator step, not global timestep)
        /*! Number of t + h == t warnings emitted by CVODE (used to cleanup output) */
        #define CV_MAX_HNIL (1)
        /*! Maximum number of CVODE error test fails before an error is thrown */
        #define CV_MAX_ERRTEST_FAILS (5) //maximum number of error test fails before an error is thrown
        
        /*! Load same initial conditions (defined in mechanism.c or mechanism.cu) for all threads */
        #define SAME_IC
            
        /*! Log output to binary file */
        #define LOG_OUTPUT
            
        /*! Turn on to log the krylov space and step sizes */
        #define LOG_KRYLOV_AND_STEPSIZES
        
        #endif
            
