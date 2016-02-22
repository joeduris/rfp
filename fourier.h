#include <iostream>
#include <vector>
#include <complex>
#include "nr3.h"

#include <math.h>
#define SWAPPER(a,b) tempr=(a);(a)=(b);(b)=tempr

using namespace std;

//void SWAP(double *a, double *b)
//	{double *dum=a; a=b; b=dum;}

//__________________________________________________________________________________________________________
double sgn(double number) {
  if(number) return number/abs(number);
  else return 1.;
}

void four1(double *data, const int n, const int isign) {
/* Replaces data[0..2*n-1] by its discrete Fourier transform, if isign is input
   as 1; or replaces data[0..2*n-1] by n times its inverse discrete Fourier transform,
   if isign is input as -1. data is a complex array of length n stored as a real array
   of length 2*n. n must be an integer power of 2. */
      
   int nn,mmax,m,j,istep,i;
   double wtemp,wr,wpr,wpi,wi,theta,tempr,tempi;
   if (n<2 || n&(n-1)) throw("n must be a power of 2 in four1");
   nn = n << 1;
   j = 1;
   for (i=1;i<nn;i+=2) {
      if (j > i) {
         SWAP(data[j-1],data[i-1]);
         SWAP(data[j],data[i]);
      }
      m=n;
      while (m >= 2 && j > m) {
         j -= m;
         m >>= 1;
      }
      j += m;
   }
   
   mmax=2;
   while (nn > mmax) {
      //cout << "fourier.h:34\tnn = " << nn << endl;
      istep=mmax << 1;
      theta=isign*(6.28318530717959/mmax);
      wtemp=sin(0.5*theta);
      wpr = -2.0*wtemp*wtemp;
      wpi=sin(theta);
      wr=1.0;
      wi=0.0;
      /*cout << "istep = " << istep << endl;
      cout << "theta = " << theta << endl;
      cout << "wtemp = " << wtemp << endl;
      cout << "wpr = " << wpr << endl;
      cout << "wpi = " << wpi << endl;*/
      //cout << "fourier.h:42" << endl;
      for (m=1;m<mmax;m+=2) {
         //cout << "fourier.h:44\tm = " << m << endl;
         for (i=m;i<=nn;i+=istep) {
            //cout << "i = " << i << "\tj = " << j << endl;
            j=i+mmax;
            tempr=wr*data[j-1]-wi*data[j];
            tempi=wr*data[j]+wi*data[j-1];
            data[j-1]=data[i-1]-tempr;
            data[j]=data[i]-tempi;
            data[i-1] += tempr;
            data[i] += tempi;
         }
         wr=(wtemp=wr)*wpr-wi*wpi+wr;
         wi=wi*wpr+wtemp*wpi+wi;
      }
      mmax=istep;  
   }
   
   // normalize
   double norm = 1. / sqrt(n);
   for(i = 0; i < 2*n; i++) {
      data[i] *= norm;
   }
}

void four1(vector<double> &data, const int isign) {
/* Overloaded interface to four1. Replaces the vector data, a complex vector of 
   length N stored as a real vector of twice that length, by its discrete Fourier
   transform, with components in wraparound order, if isign is 1; or by N times the
   inverse Fourier transform, if isign is -1. */
   four1(&data[0],data.size()/2,isign);
}

//void four1(vector< complex<double> > &data, const int isign) {
/* Overloaded interface to four1. Replaces the vector data, a complex vector of 
   length N stored as a real vector of twice that length, by its discrete Fourier
   transform, with components in wraparound order, if isign is 1; or by N times the
   inverse Fourier transform, if isign is -1. */
//   four1(&data[0],data.size()/2,isign);
//}

void fourn(double *data, vector<int> &nn, const int isign) {
/* Replaces data by its ndim-dimensional discrete Fourier transform, if isign is
   input as 1. nn[0..ndim-1] is an integer array containing the lengths of each
   dimension (number of complex values), which must all be powers of 2. data is a
   real array of length twice the product of these lengths, in which the data are
   stored as in a multidimensional complex array: real and imaginary parts of each
   element are in consecuative locations, and the rightmost index of the array
   increases most rapidly as one proceeds along data. For a two-dimensional array,
   this is equivalent to storing the array by rows. If isign is input as -1, data
   is replaced by its inverse transform times the product of the lengths of all
   dimensions.*/
   
   int idim,i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
   int ibit,k1,k2,n,nprev,nrem,ntot=1,ndim=nn.size();
   double tempi,tempr,theta,wi,wpi,wpr,wr,wtemp;
   for (idim=0;idim<ndim;idim++) ntot *= nn[idim]; // total n. of complex values
   if (ntot<2 || ntot&(ntot-1)) throw("must have powers of 2 in fourn");
   nprev=1;
   for (idim=ndim-1;idim>=0;idim--) { // main loop over the dimensions
      n=nn[idim];
      nrem=ntot/(n*nprev);
      ip1=nprev << 1;
      ip2=ip1*n;
      ip3=ip2*nrem;
      i2rev=0;
      for (i2=0;i2<ip2;i2+=ip1) {
         if (i2 < i2rev) { // this is the bit-reversal section of the routine
            for (i1=i2;i1<i2+ip1-1;i1+=2) {
               for (i3=i1;i3<ip3;i3+=ip2) {
                  i3rev=i2rev+i3-i2;
                  //cout << "i3 = " << i3 << "\ti3rev = " << i3rev << "\tdata[i3] = " << data[i3] << "\tdata[i3rev] = " << data[i3rev] << endl;
                  //cout.flush();
                  SWAP(data[i3],data[i3rev]);
                  //cout << "\ti3+1 = " << i3+1 << "\ti3rev+1 = " << i3rev+1 << "\tdata[i3+1] = " << data[i3+1] << "\tdata[i3rev+1] = " << data[i3rev+1] << endl;
                  //cout.flush();
                  SWAP(data[i3+1],data[i3rev+1]);
                  //cout << "\t." << endl;
                  //cout.flush();
               }
            }
         }
         ibit=ip2 >> 1;
         while (ibit >= ip1 && i2rev+1 > ibit) {
            i2rev -= ibit;
            ibit >>= 1;
         }
         i2rev += ibit;
      }
      ifp1=ip1;
      while (ifp1 < ip2) { // here begins the Danielson-Lanczos 
         ifp2=ifp1 << 1;   // section of the routine
         theta=isign*6.28318530717959/(ifp2/ip1); // init for the trig. recurrence
         wtemp=sin(0.5*theta);
         wpr= -2.0*wtemp*wtemp;
         wpi=sin(theta);
         wr=1.0;
         wi=0.0;
         for (i3=0;i3<ifp1;i3+=ip1) {
            for (i1=i3;i1<i3+ip1-1;i1+=2) {
               for (i2=i1;i2<ip3;i2+=ifp2) {
                  k1=i2;                        // Danielson-Lanczos formula
                  k2=k1+ifp1;
                  tempr=wr*data[k2]-wi*data[k2+1];
                  tempi=wr*data[k2+1]+wi*data[k2];
                  data[k2]=data[k1]-tempr;
                  data[k2+1]=data[k1+1]-tempi;
                  data[k1] += tempr;
                  data[k1+1] += tempi;
               }
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr;        // trig. recurrence
            wi=wi*wpr+wtemp*wpi+wi;
         }
         ifp1=ifp2;
      }
      nprev *= n;
   }
   // looks like there's a nn[0]*nn[1]*...*nn[N] factor applied each time...
   // here's a quick fix for 2D
   int datalen = 2;
   for (int i=0; i<nn.size(); i++) {
      datalen *= nn[i];
      //cout << "nn[" << i << "] = " << nn[i] << endl;
   }
   double factor = sqrt(datalen/2.);
   for (unsigned long i=0; i<datalen;i++) {
      data[i] /= factor;
   }
   //cout << "factor = " << factor << endl;
}

/*void fourn(vector<double> &data, vector<int> &nn, const int isign) {
 Overloaded version for the case where data is of type vector<double> 
   fourn(&data[0],nn,isign);
}*/

/*void fourn(double data[], unsigned long nn[], int ndim, int isign) {
 Replaces data by its ndim-dimensional discrete Fourier transform, if isign is
   input as 1. nn[0..ndim-1] is an integer array containing the lengths of each
   dimension (number of complex values), which must all be powers of 2. data is a
   real array of length twice the product of these lengths, in which the data are
   stored as in a multidimensional complex array: real and imaginary parts of each
   element are in consecuative locations, and the rightmost index of the array
   increases most rapidly as one proceeds along data. For a two-dimensional array,
   this is equivalent to storing the array by rows. If isign is input as -1, data
   is replaced by its inverse transform times the product of the lengths of all
   dimensions.
   
   int idim;
   unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
   unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
   double tempi,tempr,theta,wi,wpi,wpr,wr,wtemp;
   
   for (ntot=1,idim=1;idim<=ndim;idim++) ntot *= nn[idim]; // total n. of complex values
   //if (ntot<2 || ntot&(ntot-1)) throw("must have powers of 2 in fourn");
   nprev=1;
   for (idim=ndim;idim>=1;idim--) { // main loop over the dimensions
      n=nn[idim];
      //cout << "idim = " << idim << "\tnn[idim] = " << nn[idim] << "\tnn[0,1] = " << nn[0] << ", " << nn[1] << endl;
      //cout << "n = " << n << "\tnprev = " << nprev << "\tntot = " << ntot << endl;
      nrem=ntot/(n*nprev);
      ip1=nprev << 1;
      ip2=ip1*n;
      ip3=ip2*nrem;
      i2rev=1;
      for (i2=1;i2<=ip2;i2+=ip1) {
         if (i2 < i2rev) { // this is the bit-reversal section of the routine
            for (i1=i2;i1<=i2+ip1-2;i1+=2) {
               for (i3=i1;i3<=ip3;i3+=ip2) {
                  i3rev=i2rev+i3-i2;
                  //cout << "i3 = " << i3 << "\ti3rev = " << i3rev << "\tdata[i3] = " << data[i3] << "\tdata[i3rev] = " << data[i3rev] << endl;
                  //cout.flush();
                  SWAPPER(data[i3],data[i3rev]);
                  //cout << "\ti3+1 = " << i3+1 << "\ti3rev+1 = " << i3rev+1 << "\tdata[i3+1] = " << data[i3+1] << "\tdata[i3rev+1] = " << data[i3rev+1] << endl;
                  //cout.flush();
                  SWAPPER(data[i3+1],data[i3rev+1]);
                  //cout << "\t." << endl;
                  //cout.flush();
               }
            }
         }
         ibit=ip2 >> 1;
         while (ibit >= ip1 && i2rev > ibit) {
            i2rev -= ibit;
            ibit >>= 1;
         }
         i2rev += ibit;
      }
      ifp1=ip1;
      while (ifp1 < ip2) { // here begins the Danielson-Lanczos 
         ifp2=ifp1 << 1;   // section of the routine
         theta=isign*6.28318530717959/(ifp2/ip1); // init for the trig. recurrence
         wtemp=sin(0.5*theta);
         wpr= -2.0*wtemp*wtemp;
         wpi=sin(theta);
         wr=1.0;
         wi=0.0;
         for (i3=1;i3<=ifp1;i3+=ip1) {
            for (i1=i3;i1<=i3+ip1-2;i1+=2) {
               for (i2=i1;i2<=ip3;i2+=ifp2) {
                  k1=i2;                        // Danielson-Lanczos formula
                  k2=k1+ifp1;
                  tempr=(double)wr*data[k2]-(double)wi*data[k2+1];
                  tempi=(double)wr*data[k2+1]+(double)wi*data[k2];
                  data[k2]=data[k1]-tempr;
                  data[k2+1]=data[k1+1]-tempi;
                  data[k1] += tempr;
                  data[k1+1] += tempi;
               }
            }
            wr=(wtemp=wr)*wpr-wi*wpi+wr;        // trig. recurrence
            wi=wi*wpr+wtemp*wpi+wi;
         }
         ifp1=ifp2;
      }
      nprev *= n;
   }
}*/

//__________________________________________________________________________________________________________
double sign4(double number) {
  if(number) return number/abs(number);
  else return 1.;
}

//__________________________________________________________________________________________________________
double mod4(double number, double modulous) {
  return modulous*(number/modulous-int(number/modulous)-(sign4(number)-1)/2);
}

void prop1(double *data, const int n, double lambda, double dz, double gridwidth, int ncar) {
/* Propagate radiation stored in data. */

   double tau = 6.283185307179586;
   double tr,ti,kzdz, onebylam2=1./lambda/lambda;
   double fudgefactor=double(ncar)/double(n);
   int midpoint=(double(ncar)+1.)/2.+(n-double(ncar)-1.)/2.;
   for(int i=0; i<n; i++) {
      tr=data[2*i];
      ti=data[2*i+1];
      kzdz = dz*tau*sqrt(onebylam2 
                         - pow(fudgefactor/gridwidth*double(mod4(i-midpoint,n) - midpoint), 2.)
                         );
      data[2*i]=tr*cos(kzdz)-ti*sin(kzdz);
      data[2*i+1]=ti*cos(kzdz)+tr*sin(kzdz);
   }
}

void prop2(double *data, const int n, double lambda, double dz, double gridwidth, int ncar) {
/* Propagate radiation stored in data. */

   double tau = 6.283185307179586;
   double tr,ti,kzdz, onebylam2=1./lambda/lambda;
   double fudgefactor=1.;//double(ncar)/double(n);
   int midpoint=(double(ncar)+1.)/2.;
   if(n != ncar) midpoint += (n-double(ncar)-1.)/2.;
   for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
         tr=data[2*(n*j+i)];
         ti=data[2*(n*j+i)+1];
         kzdz = -dz*tau*sqrt(onebylam2 
                            - pow(fudgefactor/gridwidth*double(mod4(i-midpoint,n) - midpoint), 2.)
                            - pow(fudgefactor/gridwidth*double(mod4(j-midpoint,n) - midpoint), 2.)
                            );
         data[2*(n*j+i)]=tr*cos(kzdz)-ti*sin(kzdz);
         data[2*(n*j+i)+1]=ti*cos(kzdz)+tr*sin(kzdz);
      }
   }
}

// Siegman transform
void st2(double *data, const int n/*enlarged grid size*/, double lambda, double dgridin, double dgridout, double* ABDlist/*0->A, 1->B, 2->D*/, int ncar/*orig grid size*/, bool outQ = false) {
/* Propagate radiation stored in data. */

   double tau = 6.283185307179586;
   double tr,ti,phase,phasefactor,dx,scale=1.,M=dgridout/dgridin;
   
   if(outQ) {
      dx = dgridout/double(ncar-1.);
      phasefactor = (1./M-ABDlist[2])*dx*dx*tau/2./lambda/ABDlist[1];
      //scale = 1./dgridout; //for genesis, each cell is intensity so don't need this
   }
   else {
      dx = dgridin/double(ncar-1.);
      phasefactor = (M-ABDlist[0])*dx*dx*tau/2./lambda/ABDlist[1];
      //scale = dgridin; //for genesis, each cell is intensity so don't need this
   }
   
   int midpoint=(double(ncar)+1.)/2.-1;
   cout << "st2: dx = " << dx << endl;
   cout << "st2: phasefactor = " << phasefactor << endl;
   cout << "st2: midpoint = " << midpoint << endl;
   int ind = 2*(n*midpoint+midpoint); // store index for the real field component to save time
   double sqrtintens = sqrt(pow(data[ind],2.)+pow(data[ind+1],2.));
   cout << "st2: fldampmid = " << sqrtintens << endl;
   cout << "st2: intensmid = " << pow(sqrtintens,2.) << endl;
   cout << "st2: phasemid = " << acos( data[ind] / sqrtintens ) * sgn( asin( data[ind+1] / sqrtintens ) ) << endl;
   double norm = 0.;
   for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
         ind=2*(n*j+i);
         tr=data[ind];
         // NOTE: genesis has flipped sign for imaginary part of field so need to flip it before and after doing the transformation; thus, the minus sign here
         if(outQ) {ti=data[ind+1];}
         else {ti=-data[ind+1];} 
         phase = phasefactor*(pow(i-midpoint,2.)+pow(j-midpoint,2.));
         data[ind]=tr*cos(phase)-ti*sin(phase);
         // NOTE: genesis has flipped sign for imaginary part of field so need to flip it before and after doing the transformation; thus, the minus sign here
         if(outQ) {data[ind+1]=-ti*cos(phase)-tr*sin(phase);}
         else {data[ind+1]=ti*cos(phase)+tr*sin(phase);}
         // add up power
         norm += pow(tr,2.) + pow(ti,2.);
      }
   }
   cout << "st2: power = " << norm << endl;
   ind = 2*(n*midpoint+midpoint); // store index for the real field component to save time
   sqrtintens = sqrt(pow(data[ind],2.)+pow(data[ind+1],2.));
   cout << "st2: fldampmid = " << sqrtintens << endl;
   cout << "st2: intensmid = " << pow(sqrtintens,2.) << endl;
   cout << "st2: phasemid = " << acos( data[ind] / sqrtintens ) * sgn( asin( data[ind+1] / sqrtintens ) ) << endl;
}

// inverse Siegman transform
void ist2(double *data, const int n/*enlarged grid size*/, double lambda, double dgridin, double dgridout, double* ABDlist/*0->A, 1->B, 2->D*/, int ncar/*orig grid size*/) {
   return st2(data, n, lambda, dgridin, dgridout, ABDlist, ncar, true);
}

// Siegman kernel
void sk2(double *data, const int n/*enlarged grid size*/, double lambda, double dgridin, double dgridout, double* ABDlist/*0->A, 1->B, 2->D*/, int ncar/*orig grid size*/) {
/* Propagate radiation stored in data. */

   double tau = 6.283185307179586;
   double tr,ti,phase,phasefactor,Nc,f,M=dgridout/dgridin/*mag*/;
   
   Nc = M*pow(2*dgridin,2.)/lambda/ABDlist[1]; // collimated Fresnel number 
   f=pow(double(ncar)/double(n),2.); // fudge factor for when n!=ncar 
   cout << "sk2: Nc = " << Nc << endl;
   cout << "sk2: f = " << f << endl;
   cout << "sk2: f*Nc = " << f*Nc << endl;
   
   phasefactor = tau*f/2./Nc;
   
   double midpt=double(n)/2.; 
   cout << "sk2: phasefactor = " << phasefactor << endl;
   cout << "sk2: midpt = " << midpt << endl;
   int ind = 0; // store index for the real field component to save time
   double sqrtintens = sqrt(pow(data[ind],2.)+pow(data[ind+1],2.));
   cout << "sk2: fldampmid = " << sqrtintens << endl;
   cout << "sk2: intensmid = " << pow(sqrtintens,2.) << endl;
   cout << "sk2: phasemid = " << acos( data[ind] / sqrtintens ) * sgn( asin( data[ind+1] / sqrtintens ) ) << endl;
   for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
         ind=2*(n*j+i);
         tr=data[ind];
         ti=data[ind+1];
         phase = phasefactor*(pow(mod4(i-midpt,n)-midpt,2.)+pow(mod4(j-midpt,n)-midpt,2.));
         data[ind]=tr*cos(phase)-ti*sin(phase);
         data[ind+1]=ti*cos(phase)+tr*sin(phase);
      }
   }
   ind = 0;
   sqrtintens = sqrt(pow(data[ind],2.)+pow(data[ind+1],2.));
   cout << "sk2: fldampmid = " << sqrtintens << endl;
   cout << "sk2: intensmid = " << pow(sqrtintens,2.) << endl;
   cout << "sk2: phasemid = " << acos( data[ind] / sqrtintens ) * sgn( asin( data[ind+1] / sqrtintens ) ) << endl;
}

