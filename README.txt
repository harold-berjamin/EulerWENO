Finite volume WENO code for the 1D compressible Euler equations of fluid dynamics.

It was initially a matlab tool that was turned into python (cf. folders).

A. To use the python code:
1. Open 'euler_hb.py', the main file.
2. In the Configuration section, you can import a pre-defined test
   (see section B. below). Otherwise, you can set custom parameters
   in the dedicated section (to uncomment blockwise).
3. You can set the graphics output 'plots', where
	0 does not produce graphics
	1 displays density
	2 displays momentum
	3 displays energy
4. You can set the mesh size 'Nx' and Courant number 'Co' below.


B. To run a pre-defined test
1. Open 'euler_tests.py', which is a configuration file.
2. Unless needed, leave the common parameters unchanged.
3. Select the desired configuration 'test' whose parameters are shown below
   (to be left unchanged unless needed).
4. For the selection of the Riemann data (test 2),
   set the custom Riemann data similarly to the Lax & Sod examples above
   (to be left unchanged unless needed).


C. To perform error measurements
1. Select first test in 'euler_tests.py' and run the simulation.
2. Errors are displayed in the terminal.