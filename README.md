# rfp - radiation field propagator for genesis dfl field files

### Python version
- rfp.py - Fresnel propagation and plotting functions; optionally parallelized with pyfftw for faster compulations;
- dfl_prop - Script taking arguments to propagate dfl files
- dfl_plot - Script taking arguments to plot dfl files
- dfl_prop_script_cbxfel.py - Script which propagates dfl files through a rectangular cavity (9.8 keV) with 4 Bragg diamond mirrors and two lenses. Options to save plots of fields after each optical element. Optionally reduces some unnecessary FFTs for speedups (messes up intermediate plots).
- Bragg_mirror.py - Bragg mirror code ported to python from Gabe Marcus' matlab function based on https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.15.100702

### C++ version
- rfp.cpp - Program for Fresnel propagation. Compile with: g++ rfp.cpp -o rfp
- rfcp.cpp - Program for rectangular cavity Fresnel propagation. Compile with: g++ rfcp.cpp -o rfcp
- fourier.h - Dependency for above: simple fft code from numerical recipes.
