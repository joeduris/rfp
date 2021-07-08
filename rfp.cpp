/*
Genesis radiation field propagator (rfp)
Contact: Joe Duris - jduris@physics.ucla.edu

to compile: g++ rfp.cpp -o rfp

based on field phase flipper (fpf) program
based on code dfl parsing code from Genesis Informed Tapering Scheme (gits)

2015-10-29 Started project (currently only for 1 slice; time independent mode)
2015-11-04 Using Numerical Recipe's fourn to do the ffts.
           Added the point spread function to fourier.h
2016-01-06 Using the Siegman transform (Siegman section 20.6) that Pietro found
           Added functions st2, ist2, and sk2 to fourier.h
2017-01-17 Looks like rhel6 g++ doesn't know uint64_t :/ manually defining it
2016-01-19 Adding a mask to cut the tails
2018-05-21 Adding time dependence (unfortunately have to change the arguments order)
2018-05-22 Adding slippage and filling in absent slices with zeros
2018-06-14 Flipped region for cutradius from outside to inside
*/

// headers taken blindly from gits so a few might be useless
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#include "fourier.h"

typedef unsigned long long int   uint64_t; // i hate rhel6

using namespace std;

/* moved to fourier.h
//__________________________________________________________________________________________________________
double sgn(double number) {
  if(number) return number/abs(number);
  else return 1.;
}*/

//__________________________________________________________________________________________________________
int radfieldprop(string dflfilename, string dflofilename, double xlamds, double dgridin, double dgridout, double A = 1., double B = 0., double D = 1., double intensity_scale_factor = 1., int ncar = 0, int nslip = 0, int verbose = 0, double cutradius = 0.) {

   // field amplitude scale factor
   double field_scale_factor = sqrt(intensity_scale_factor);
   
   // ABCD matrix
   double ABDlist [] = {A, B, D};

   // open input dfl for reading
   ifstream dfli(dflfilename.c_str(), ios::binary );
   if(!dfli) {
      cout << "ERROR: Could not open file " << dflfilename << endl;
      return -1;
   }
   
   // open output dfl for writing
   //string dflofilename = dflfilename + ".rfp";
   ofstream dflo(dflofilename.c_str(), ios::binary );
   if(verbose) cout << "Writing to " << dflofilename << endl;
   
   // check filesize
   dfli.seekg(0, dfli.end);
   uint64_t fileSize_dfl = dfli.tellg();
   if(verbose) cout << "INFO: File size of " << dflfilename << " is " << fileSize_dfl << endl;
      
   // seek beginning of file
   dfli.seekg(0, dfli.beg);
   
   /* debug
   cout << "fileSize_dfl = " << fileSize_dfl << endl;
   cout << "nfill = " << nfill << endl;
   cout << "nremain = " << nremain << endl;
   */
   
   // figure out geometry
   int nslice = 1;
   if(ncar <= 1) ncar = (int) sqrt(fileSize_dfl / 2 / sizeof(double)); // only works for time indep (nslice = 1)
   else nslice = int(fileSize_dfl / ncar / ncar / 2 / sizeof(double));
   
   //double rad[2][ncar][ncar];
   int ftgridsize = pow(2.,ceil(log(ncar)/log(2.))); // note: must round up to nearest power of 2 for fft
   if(verbose) cout << "ncar = " << ncar << "\tftgridsize = " << ftgridsize << "\tnslice = " << nslice << "\tnslip = " << nslip << "\tdgridin = " << dgridin << "\tdgridout = " << dgridout << endl;
   
   // storage for fields; let's process the data one slice at a time for memory considerations
   long ssize = 2*ftgridsize*ftgridsize; // slice size
   double rad[ssize];
   vector<int> len; len.push_back(ftgridsize); len.push_back(ftgridsize); // for fft (see below)
   
   // more geometry
   int midcell = (ncar - 1) / 2;
   int midcella = 2 * midcell + 2 * ftgridsize * midcell;
   double cutradius2 = cutradius * cutradius;
   double sqrtintens, intens, sqrtintens0, radphase, phasemid, phasemid0, radphaseshift; // for measuring on axis phase
   double phaseavg[nslice]; for(int s = 0; s < nslice; s++) phaseavg[s] = 0; // zero array
   double intensitySum[nslice]; for(int s = 0; s < nslice; s++) intensitySum[s] = 0; // zero array
   double sumIntensityWeightedPhases = 0., sumIntensities = 0., meanIntensityWeightedPhase = 0.;
   double sumAllIntensityWeightedPhases = 0., sumAllIntensities = 0., meanAllIntensityWeightedPhase = 0.;
   int ir, ii; // holder for indices
   
   // first measure intensity weighted average phase
   
   bool shiftOnAxisPhaseQ = true; // otherwise average phases <<--------------
   
   bool identifyTransform = false; // rfp cannot do identity transform so skip that processing
   if(A == 1. && B == 0. && D == 1.) {
       identifyTransform = true;
       cout << "WARNING: Transport matrix is identity matrix so skipping Fourier transforms. The output dgrid will be the same as the input dgrid!" << endl;
   }
   
   if(identifyTransform) {
       radphaseshift = 0.;
   }
   else {
   
        if(verbose) cout << endl << "Reading phases for transformed slices first." << endl << endl;
        
        // begin the loop over slices (save memory by processing one slice at a time)
        for(int s = 0; s < nslice; s++) {
            
        //        if(verbose) cout << endl << "Reading phase for slice " << s + 1 << endl << endl;
            
                // zero the array (for the margins)
                for(int i = 0; i < 2*ftgridsize*ftgridsize; i++) {
                    rad[i] = 0.;
                }
                
                // read that shit
                for(int y = 0; y < ncar; y++) {// columns
                    for(int x = 0; x < ncar; x++) {// rows
                        for(int i = 0; i < 2; i++) {// components
                            dfli.read((char*) &rad[i+2*x+2*ftgridsize*y], sizeof(double));
                        }
                    }
                }
                
                // mask the input radiation 
                if(cutradius != 0.) {
                    if(verbose) cout << "cutting radiation less than " << cutradius << " cells from the grid center" << endl;
                    for(int y = 0; y < ncar; y++) {// columns
                        for(int x = 0; x < ncar; x++) {// rows
                            if(pow(x-midcell,2.) + pow(y-midcell,2.) <= cutradius2) {
                                ir = 2*x+2*ftgridsize*y; // relative address of real component
                                rad[ir] = 0.;
                                rad[ir + 1] = 0.;
                            }
                        }
                    }
                }
                
                // siegman transform
        //         if(verbose) cout << "st2" << endl;
                st2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
                
                // fourier transform
                fourn(rad, len, 1);
                
                // siegman kernel
        //         if(verbose) cout << "sk2" << endl;
                sk2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
                
                // inverse fourier transform
                fourn(rad, len, -1);
                
                // siegman kernel
        //         if(verbose) cout << "ist2" << endl;
                ist2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
                
                // measure on-axis phase
                sqrtintens = sqrt(pow(rad[midcella], 2.) + pow(rad[midcella+1], 2.));
                if(sqrtintens == 0.) phasemid = 0.;
                else phasemid = acos( rad[midcella] / sqrtintens ) * sgn( asin( rad[midcella+1] / sqrtintens ) );
                
                // sum intensity weighted phases
                sumIntensityWeightedPhases = 0.; sumIntensities = 0.;        
                if(shiftOnAxisPhaseQ) { // use the center pixel for a phase target
                    // calculate sums
                    sumIntensities = sqrtintens * sqrtintens;
                    sumIntensityWeightedPhases = phasemid * sumIntensities;
                }
                else { // intensity weighted phase over a slice
                    for(int y = 0; y < ncar; y++) {// columns
                        for(int x = 0; x < ncar; x++) {// rows
                            ir = 2*x+2*ftgridsize*y; // relative address of real component
                            ii = ir + 1; // relative address of the imaginary component
                            if(pow(x-midcell,2.) + pow(y-midcell,2.) > cutradius2 || cutradius == 0.) {
                                intens = pow(rad[ir], 2.) + pow(rad[ii], 2.);
                                if(intens == 0.)
                                    continue; // avoid divide by zero
                                else {
                                    sqrtintens = sqrt(intens);
                                    radphase = acos( rad[ir] / sqrtintens ) * sgn( asin( rad[ii] / sqrtintens ) );
                                    sumIntensityWeightedPhases += intens * radphase;
                                    sumIntensities += intens;
                                }
                                if(radphase != radphase) {
                                    cout << "ERROR @ [" << x << ", " << y << "]: " << radphase << "\t"
                                        << intens << "\t" << sqrtintens << ",\t" 
                                        << rad[ir] << "\t" << rad[ii] << ",\t" 
                                        << rad[ir] / sqrtintens << "\t" << rad[ii] / sqrtintens << endl;
                                }
                            }
                        }
                    }
                }
                    
                // calculate sums
                sumAllIntensityWeightedPhases += sumIntensityWeightedPhases;
                sumAllIntensities += sumIntensities;
                if(sumIntensities)
                    meanIntensityWeightedPhase = sumIntensityWeightedPhases / sumIntensities;
                else
                    meanIntensityWeightedPhase = 0.;
                phaseavg[s] = meanIntensityWeightedPhase;
                intensitySum[s] = sumIntensities;
                
                // report
                if(verbose) cout << "INFO: slice " << s << ": phasemid = " << phasemid << "\tphaseavg = " << phaseavg[s] << "\tintensitySum = " << intensitySum[s] << endl;
                
        } // end loop over slices
        
        // calculate intesity weighted average phase
        meanAllIntensityWeightedPhase = sumAllIntensityWeightedPhases / sumAllIntensities;
        radphaseshift = -meanAllIntensityWeightedPhase;
        if(verbose) {
            cout << "INFO: meanAllIntensityWeightedPhase = " << meanAllIntensityWeightedPhase << endl;
            cout << "INFO: radphaseshift = " << radphaseshift << endl;
        }
   }
   
   // Now repeat, but this time shift phases
       
   // seek beginning of file
   dfli.seekg(0, dfli.beg);
   
   /* how to get the slice ordering right with slippage (Mathematica)
   nslice=10;nslip=9;
   r1={#,#[[1]]<#[[2]]}&@{0,nslip}
   r2={#,#[[1]]<#[[2]]}&@{nslice-Min[nslice+nslip,nslice],nslice-Abs[Min[0,-nslip]]}
   r3={#,#[[1]]>#[[2]]}&@{0,nslip} (*note the greater than symbol*)
   Length[Flatten[{Table[,{i,#[[1]],#[[2]],1}]&[r1[[1]]],Table[,{i,#[[1]],#[[2]],1}]&[r2[[1]]],Table[,{i,#[[1]],#[[2]],-1}]&[r3[[1]]]}]]-2 */
   
   // write empty slices at tail if any
   double zero = 0.;
   for(int s = 0; s < nslip; s++) {
        if(verbose) cout << endl << "Adding trailing null slice " << s + 1 << endl << endl;
        // write
        for(int y = 0; y < ncar; y++) // columns
            for(int x = 0; x < ncar; x++) // rows
                for(int i = 0; i < 2; i++) // components
                    dflo.write((char*) &zero, sizeof(double));
   }
   
   // begin the loop over slices (save memory by processing one slice at a time)
   for(int s = 0; s < nslice - abs(min(0,-nslip)); s++) {
       
        // zero the array (for the margins)
        for(int i = 0; i < 2*ftgridsize*ftgridsize; i++) {
            rad[i] = 0.;
        }
        
        // read that shit
        for(int y = 0; y < ncar; y++) {// columns
            for(int x = 0; x < ncar; x++) {// rows
                for(int i = 0; i < 2; i++) {// components
                    dfli.read((char*) &rad[i+2*x+2*ftgridsize*y], sizeof(double));
                }
            }
        }
        
        if(s < nslice-min(nslice+nslip,nslice)) // need to skip some leading slices
            continue;
       
        if(verbose) cout << endl << "Processing slice " << s + 1 << endl << endl;
        
        // mask the input radiation
        if(cutradius != 0.) {
            if(verbose) cout << "cutting radiation less than " << cutradius << " cells from the grid center" << endl;
            for(int y = 0; y < ncar; y++) {// columns
                for(int x = 0; x < ncar; x++) {// rows
                    if(pow(x-midcell,2.) + pow(y-midcell,2.) <= cutradius2) {
                        ir = 2*x+2*ftgridsize*y; // relative address of real component
                        rad[ir] = 0.; // real part
                        rad[ir + 1] = 0.; // imaginary part
                    }
                }
            }
        }
        
        if(!identifyTransform) { // if the transport matrix is the identity, then skip the transform
        
            // measure on-axis phase
            sqrtintens = sqrt(pow(rad[midcella], 2.) + pow(rad[midcella+1], 2.));
            if(sqrtintens == 0.) phasemid = 0.;
            else phasemid = acos( rad[midcella] / sqrtintens ) * sgn( asin( rad[midcella+1] / sqrtintens ) );
            
            if(verbose) {
                cout << "before transform: fnormmid = " << sqrtintens 
                                    << "\tintensmid = " << pow(sqrtintens, 2.)
                                    << "\tphasemid = " << phasemid << endl;
            }
            
            // siegman transform
    //         if(verbose) cout << "st2" << endl;
            st2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
            
            // fourier transform
            fourn(rad, len, 1);
            
            // siegman kernel
    //         if(verbose) cout << "sk2" << endl;
            sk2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
            
            // inverse fourier transform
            fourn(rad, len, -1);
            
            // siegman kernel
    //         if(verbose) cout << "ist2" << endl;
            ist2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);//, verbose);
            
            // measure on-axis phase
            sqrtintens = sqrt(pow(rad[midcella], 2.) + pow(rad[midcella+1], 2.));
            if(sqrtintens == 0.) phasemid = 0.;
            else phasemid = acos( rad[midcella] / sqrtintens ) * sgn( asin( rad[midcella+1] / sqrtintens ) );
            
            if(verbose) {
                cout << "after transform: fnormmid = " << sqrtintens
                                    << "\tintensmid = " << pow(sqrtintens, 2.)
                                    << "\tphasemid = " << phasemid
                                    << "\tradphaseshift = " << radphaseshift << endl;
            }
            
            // shift the phases
            for(int y = 0; y < ncar; y++) {// columns
                for(int x = 0; x < ncar; x++) {// rows
                    ir = 2*x+2*ftgridsize*y;
                    ii = ir + 1;
                    sqrtintens = sqrt(pow(rad[ir], 2.) + pow(rad[ii], 2.));
                    if(sqrtintens == 0.) continue; // avoid divide by zero // radphase = radphaseshift; 
                    radphase = radphaseshift + acos( rad[ir] / sqrtintens ) * sgn( asin( rad[ii] / sqrtintens ) );
                    //cout << x << "\t" << y << "\t" << rad[0][x][y] << "\t" << rad[1][x][y] << "\t" << sqrtintens << "\t" << radphase << "\t";
                    sqrtintens *= field_scale_factor; // scale the field
                    rad[ir] = sqrtintens * cos(radphase);
                    rad[ii] = sqrtintens * sin(radphase);
                    //cout << rad[0][x][y] << "\t" << rad[1][x][y] << endl;
                }
            }
            
            if(verbose) {
                sqrtintens = sqrt(pow(rad[midcella], 2.) + pow(rad[midcella+1], 2.));
                if(sqrtintens == 0.) phasemid = 0.;
                else phasemid = acos( rad[midcella] / sqrtintens ) * sgn( asin( rad[midcella+1] / sqrtintens ) );
                cout << "after phase shift & intensity scale: fnormmid = " << sqrtintens << "\t"
                    << "intensmid = " << pow(sqrtintens, 2.) << "\t"
                    << "phasemid = " << phasemid << endl;
            }
        }
        
        // write
        for(int y = 0; y < ncar; y++) // columns
            for(int x = 0; x < ncar; x++) // rows
                for(int i = 0; i < 2; i++) // components
                    dflo.write((char*) &rad[i+2*x+2*ftgridsize*y], sizeof(double));
                
   } // end loop over slices: s = 0..nslice
   
   // write empty slices at tail if any
   for(int s = 0; s > nslip; s--) {
        if(verbose) cout << endl << "Adding leading null slice " << s + 1 << endl << endl;
        // write
        for(int y = 0; y < ncar; y++) // columns
            for(int x = 0; x < ncar; x++) // rows
                for(int i = 0; i < 2; i++) // components
                    dflo.write((char*) &zero, sizeof(double));
   }
   
   // close files
   dfli.close(); // input file
   dflo.close(); // output file
   
   if(identifyTransform) {
       cout << "WARNING: Transport matrix is identity matrix so skipped Fourier transforms. The output dgrid is the same as the input dgrid!" << endl;
   }
   
   return 0;
   
}


//__________________________________________________________________________________________________________
int main(int argc, char *argv[]) {
   
   // NOTE: we can change the dgrid if desired but not implemented here

   // don't run without arguments
   if(argc < 9 || argc > 14) {
      cout << "rfp - Radiation Field Propagator propagates radiation from a Genesis dfl radiation file" << endl;
      cout << "Usage: " << argv[0] << " input_dfl_filename output_dfl_filename xlamds dgridin A B D [intensity_scale_factor] [ncar] [nslip] [verboseQ] [cutradius] [dgridout]" << endl;
      cout << "Note: ncar (Genesis grid size) is needed for time dependence to figure out how many slices to process." << endl;
//       cout << "rfcp - Radiation Field Cavity Propagator propagates radiation from a Genesis dfl radiation file" << endl;
//       cout << "Usage: " << argv[0] << " input_dfl_filename output_dfl_filename xlamds dgrid Lu Lc dl df [intensity_scale_factor] [ncar] [nslip] [verboseQ] [cutradius]" << endl;
//       cout << "Note: Lu is undulator length, Lc is length of the cavity, dl moves the lens from the center of the cavity, and df moves the focal length about a default Lc/4." << endl;
//       cout << "Note: ncar (Genesis grid size) is needed for time dependence to figure out how many slices to process." << endl;
      cout << "Note: Optional integer argument nslip < 0 slips the radiation back by that many slices." << endl;
      return -1;
   }
   
   // declare arguments
   string input_dfl_filename, output_dfl_filename;
   double intensity_scale_factor = 1., xlamds, dgridin, dgridout = -1., A, B, D, Lu, Lc, dl, df, cutradius = 0;
   int verboseQ = 0, ncar = 0, iarg = 1, nslip = 0;
   
   // parse arguments
   int narglim = 8;
   if(argc >= narglim++) {
      input_dfl_filename = argv[iarg++];
      output_dfl_filename = argv[iarg++];
      xlamds = strtod(argv[iarg++],0);
      dgridin = strtod(argv[iarg++],0);
      A = strtod(argv[iarg++],0);
      B = strtod(argv[iarg++],0);
      D = strtod(argv[iarg++],0);
   }
   if(argc >= narglim++) {
      intensity_scale_factor = strtod(argv[iarg++],0);
   }
   if(argc >= narglim++) {
      ncar = int(strtod(argv[iarg++],0));
   }
   if(argc >= narglim++) {
      nslip = int(strtod(argv[iarg++],0));
   }
   if(argc >= narglim++) {
      verboseQ = strtod(argv[iarg++],0);
   }
   if(argc >= narglim++) {
      cutradius = strtod(argv[iarg++],0); // in units of the grid spacing
   }
   if(argc >= narglim++) {
      dgridout = strtod(argv[iarg++],0); // in units of the grid spacing
   }
   
   // calculate the ABCD matrix elements (note: determinant constrains elements so C isn't needed)
//    A = (4*df + 4*dl - Lc + 2*Lu)/(4*df + Lc);
//    B = (4*dl*dl + (Lc - Lu)*(4*df + Lu))/(4*df + Lc);
//    D = (4*df - 4*dl - Lc + 2*Lu)/(4*df + Lc);
   
   if(verboseQ) { 
       cout << "xlamds = " << xlamds << endl;
       cout << "dgridin  = " << dgridin << endl;
       cout << "intensity_scale_factor  = " << intensity_scale_factor << endl;
       if(ncar) cout << "ncar = " << ncar << endl;
       else cout << "ncar = automatic" << endl;
       cout << "A = " << A << endl;
       cout << "B = " << B << endl;
       cout << "D = " << D << endl;
       cout << "nslip = " << nslip << endl;
       cout << "cutradius = " << cutradius << endl;
       cout << "dgridout  = " << dgridout << endl;
   }
   
   // same as above?
   /*A = (df + dl - 0.25*Lc + 0.5*Lu)/(df + 0.25*Lc);
   B = (dl*dl + (Lc - Lu)*(df + 0.25*Lu))/(df + 0.25*Lc);
   D = (df - dl - 0.25*Lc + 0.5*Lu)/(df + 0.25*Lc);*/
   
   if(dgridout < 0) dgridout = dgridin;
   
   // do a single file
   return radfieldprop(input_dfl_filename, output_dfl_filename, xlamds, dgridin, dgridout, A, B, D, intensity_scale_factor, ncar, nslip, verboseQ, cutradius);
   
}

