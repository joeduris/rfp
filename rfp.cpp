// Genesis radiation field propagator (rfp)
// Contact: Joe Duris - jduris@physics.ucla.edu

// to compile: g++ rfp.cpp -o rfp

// based on field phase flipper (fpf) program
// based on code dfl parsing code from Genesis Informed Tapering Scheme (gits)

// 2015-10-29 Started project (currently only for 1 slice; time independent mode)
// 2015-11-04 Using Numerical Recipe's fourn to do the ffts.
//            Added the point spread function to fourier.h
// 2016-01-06 Using the Siegman transform (Siegman section 20.6) that Pietro found
//            Added functions st2, ist2, and sk2 to fourier.h


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

using namespace std;

/* moved to fourier.h
//__________________________________________________________________________________________________________
double sgn(double number) {
  if(number) return number/abs(number);
  else return 1.;
}*/

//__________________________________________________________________________________________________________
int radfieldprop(string dflfilename, double xlamds, double dgridin, double dgridout, double A = 1., double B = 0., double D = 1., double intensity_scale_factor = 1., int verbose = 0) {

   // field amplitude scale factor
   double field_scale_factor = sqrt(intensity_scale_factor);

   // open input dfl for reading
   ifstream dfli(dflfilename.c_str(), ios::binary );
   if(!dfli) {
      cout << "ERROR: Could not open file " << dflfilename << endl;
      return -1;
   }
   
   // not going to worry about nslices or ncar, but check filesize
   dfli.seekg(0, dfli.end);
   uint64_t fileSize_dfl = dfli.tellg();
   if(verbose) cout << "INFO: File size of " << dflfilename << " is " << fileSize_dfl << endl;
   
   /* debug
   cout << "fileSize_dfl = " << fileSize_dfl << endl;
   cout << "nfill = " << nfill << endl;
   cout << "nremain = " << nremain << endl;
   */
   
   // prep for data import
   int ncar = (int) sqrt(fileSize_dfl / 2 / sizeof(double)); // only works for time indep (nslice = 1)
   //double rad[2][ncar][ncar];
   int ftgridsize = pow(2.,ceil(log(ncar)/log(2.)));
   if(verbose) cout << "ncar = " << ncar << "\tftgrid = " << ftgridsize << endl;
   double rad[2*ftgridsize*ftgridsize];
      
   // seek beginning of file
   dfli.seekg(0, dfli.beg);
   
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
   // close file
   dfli.close();
   
   // measure on-axis phase
   
   int midcell = (ncar - 1) / 2;
   midcell = 2 * midcell + 2 * ftgridsize * midcell;
   double sqrtintens = sqrt(pow(rad[midcell], 2.) + pow(rad[midcell+1], 2.));
   if(verbose) {
      cout << "before transform:" << endl;
      cout << "fnormmid = " << sqrtintens << endl;
      cout << "intensmid = " << pow(sqrtintens, 2.) << endl;
      cout << "phasemid = " << acos( rad[midcell] / sqrtintens ) * sgn( asin( rad[midcell+1] / sqrtintens ) ) << endl;
   }
   
   
   // ABCD matrix
   
   double ABDlist [] = {A, B, D};
   
   // siegman transform
   st2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);
   
   // fourier transform
   vector<int> len;
   len.push_back(ftgridsize);
   len.push_back(ftgridsize);
   fourn(rad, len, 1);
   
   // siegman kernel
   sk2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);
   
   // inverse fourier transform
   fourn(rad, len, -1);
   
   // siegman kernel
   ist2(rad, ftgridsize, xlamds, dgridin, dgridout, ABDlist/*0->A, 1->B, 2->D*/, ncar/*orig grid size*/);
   
   // measure on-axis phase
   //int 
   midcell = (ncar - 1) / 2;
   //midcell += ftgridsize * midcell;
   midcell = 2*midcell+2*ftgridsize*midcell;
   //cout << "(ncar + 1) / 2 = " << (ncar + 1) / 2 << "\t midcell = " << midcell << endl;
   //double 
   sqrtintens = sqrt(pow(rad[midcell], 2.) + pow(rad[midcell+1], 2.));
   double radphaseshift;
   if(sqrtintens == 0.) radphaseshift = 0.;
   else radphaseshift = -acos( rad[midcell] / sqrtintens ) * sgn( asin( rad[midcell+1] / sqrtintens ) );
   
   if(verbose) {
      cout << "after transform:" << endl;
      cout << "fnormmid = " << sqrtintens << endl;
      cout << "intensmid = " << pow(sqrtintens, 2.) << endl;
      cout << "phasemid = " << acos( rad[midcell] / sqrtintens ) * sgn( asin( rad[midcell+1] / sqrtintens ) ) << endl;
      cout << "radphaseshift = " << radphaseshift << endl;
   }
   
   
   // shift the phase
   
   double radphase;
   int ir, ii; // holder for indices
   for(int y = 0; y < ncar; y++) {// columns
      for(int x = 0; x < ncar; x++) {// rows
         ir = 2*x+2*ftgridsize*y;
         ii = ir + 1;
         sqrtintens = sqrt(pow(rad[ir], 2.) + pow(rad[ii], 2.));
         if(sqrtintens == 0.) radphase = radphaseshift;
         else radphase = radphaseshift + acos( rad[ir] / sqrtintens ) * sgn( asin( rad[ii] / sqrtintens ) );
         //cout << x << "\t" << y << "\t" << rad[0][x][y] << "\t" << rad[1][x][y] << "\t" << sqrtintens << "\t" << radphase << "\t";
         sqrtintens = field_scale_factor * sqrtintens; // scale the field
         rad[ir] = sqrtintens * cos(radphase);
         rad[ii] = sqrtintens * sin(radphase);
         //cout << rad[0][x][y] << "\t" << rad[1][x][y] << endl;
      }
   }
   
   if(verbose) {
      cout << "after phase shift and intensity scale:" << endl;
      sqrtintens = sqrt(pow(rad[midcell], 2.) + pow(rad[midcell+1], 2.));
      cout << "fnormmid = " << sqrtintens << endl;
      cout << "intensmid = " << pow(sqrtintens, 2.) << endl;
      cout << "phasemid = " << acos( rad[midcell] / sqrtintens ) * sgn( asin( rad[midcell+1] / sqrtintens ) ) << endl;
   }
   
   
   
   // open output dfl for writing
   string dflofilename = dflfilename + ".rfp";
   ofstream dflo(dflofilename.c_str(), ios::binary );
   if(verbose) cout << "Writing to " << dflofilename << endl;
   
   // write
   for(int y = 0; y < ncar; y++) // columns
      for(int x = 0; x < ncar; x++) // rows
         for(int i = 0; i < 2; i++) // components
            dflo.write((char*) &rad[i+2*x+2*ftgridsize*y], sizeof(double));
   
   // close file
   dflo.close();
   
   
   return 0;
   
}


//__________________________________________________________________________________________________________
int main(int argc, char *argv[]) {

   // don't run without arguments
   if(argc != 8 && argc != 9 && argc != 10) {
      cout << "rfp - Radiation Field Propagator propagates radiation from a Genesis dfl radiation file" << endl;
      cout << "Usage: " << argv[0] << " input_dfl_filename xlamds dgridin dgridout A B D [intensity_scale_factor] [verboseQ]" << endl;
      cout << "Note: A, B, and D are elements of an ABCD transfer matrix." << endl;
      return -1;
   }
   
   // declare arguments
   string input_dfl_filename;
   double intensity_scale_factor, xlamds, dgridin, dgridout, A, B, D;
   int verboseQ;
   if(argc == 8) {
      input_dfl_filename = argv[1];
      xlamds = strtod(argv[2],0);
      dgridin = strtod(argv[3],0);
      dgridout = strtod(argv[4],0);
      A = strtod(argv[5],0);
      B = strtod(argv[6],0);
      D = strtod(argv[7],0);
      intensity_scale_factor = 1.;
      verboseQ = 0;
   }
   if(argc == 9) {
      input_dfl_filename = argv[1];
      xlamds = strtod(argv[2],0);
      dgridin = strtod(argv[3],0);
      dgridout = strtod(argv[4],0);
      A = strtod(argv[5],0);
      B = strtod(argv[6],0);
      D = strtod(argv[7],0);
      intensity_scale_factor = strtod(argv[8],0);
      verboseQ = 0;
   }
   if(argc == 10) {
      input_dfl_filename = argv[1];
      xlamds = strtod(argv[2],0);
      dgridin = strtod(argv[3],0);
      dgridout = strtod(argv[4],0);
      A = strtod(argv[5],0);
      B = strtod(argv[6],0);
      D = strtod(argv[7],0);
      intensity_scale_factor = strtod(argv[8],0);
      verboseQ = strtod(argv[9],0);
   }
   
   // do a single file
   return radfieldprop(input_dfl_filename, xlamds, dgridin, dgridout, A, B, D, intensity_scale_factor, verboseQ);
      
}

