# -*- coding: iso-8859-1 -*-

# rfp - radiation field propagator
# tools for manipulating genesis dfl field files
# J. Duris jduris@slac.stanford.edu

# Bragg mirror code ported to python from Gabe Marcus' matlab function based on https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.15.100702


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def Bragg_mirror(photon_energies_eV, angles_rad, reflectionQ=True, undo_slippageQ=True, undo_xshiftQ=True):
    
    e_charge = 1.602176565e-19;     # charge unit[C]
    # h_Plank  = 6.62607004e-34;      # Plank constant [J-sec]
    h_Plank = 4.135667696e-15;      # Plank constant [eV-sec]
    c_speed  = 299792458;           # speed of light[m/sec]
    pi = np.pi

    ####### Crystal electric susceptibility data from XOP ######
    chi_0  = -0.151228E-04 + 1j*0.146454E-07;   # Diamond electric susceptibility vacuum
    chi_h = 0.372253E-05 - 1j*0.141091E-07;
    chi_hbar = chi_h; # for centro-symmetric crystal
    ############################################################

    d = 500e-6; #crystal thickness
    
    eph = photon_energies_eV
    #eph = np.linspace(9831.65,9832.25,5e3+1); # photon energy in eV over which to evaluate the system
    lambd = h_Plank*c_speed/eph;
    theta_0 = 45.0*pi/180.; # nominal angle of incidence, most likely equal to the Bragg angle for the target photon energy
    #dtheta_0 = 50e-6; # range of theta to interate over
    #ntheta_0 = 5e3+1;
    #theta = np.linspace(theta_0-dtheta_0,theta_0+dtheta_0,ntheta_0); # angle of incidence in radians over which to evaluate the system
    theta = angles_rad # + theta_0 # make sure to center the angles on 45 deg
    
    Lambda,Theta = np.meshgrid(lambd,theta);
    
    eta = 0.0; # asymetry angle
    gamma_0 = np.cos(Theta + eta - pi/2.);
    gamma_h = np.cos(pi/2. + Theta - eta);
    b = gamma_0/gamma_h; # asymmetry factor
    P = 1.0; # polarization factor, sigma case = 1

    dh = 0.89170487e-10;#0.891704865e-10; #interplanar distance (d-spacing?) (for 400 diamond at 9831 eV for 45 degree bragg?)
    H = 2.*pi/dh; # Bragg vector
    K0 = 2.*pi/Lambda; # magnitude of K0 vector
    # alpha = (2*K0*H*np.sin(Theta)-H**2)/(K0**2); #Deviation parameter
    alpha = (H/K0)*(H/K0-2.*gamma_0);

    lambda_hs = np.sin(Theta)/(K0*np.abs(P)*np.sqrt(chi_h*chi_hbar)); # symmetric extinction length
    lambda_h = np.sqrt(gamma_0*np.abs(gamma_h))/(K0*np.abs(P)*np.sqrt(chi_h*chi_hbar)); # extinction length
    y = K0*lambda_h/(2.*gamma_0)*(b*alpha+chi_0*(1.-b));

    Y1 = -y+np.sqrt(y**2+b/np.abs(b));
    Y2 = -y-np.sqrt(y**2+b/np.abs(b));
    G = np.sqrt(np.abs(b)*chi_h*chi_hbar)/chi_hbar;
    R1 = G*Y1;
    R2 = G*Y2;
    A = d/lambda_h;
    
    if reflectionQ:
        R0H = R1*R2*(1-np.exp(1j*A/2*(Y1-Y2)))/(R2-R1*np.exp(1j*A/2*(Y1-Y2)));       #   bragg diffraction amplitude
        
        if undo_slippageQ:
            ds_shift = 2. * np.abs(chi_h) * np.sin(theta_0)
            ds_shift *= (1./np.sin(theta_0)+1.)/2. # fudge factor to reduce slippage more
            R0H *= np.exp(-1j * 2. * np.pi / Lambda * ds_shift)
        if undo_xshiftQ:
            dx_shift = 2. * np.abs(chi_h) * np.cos(theta_0)
            dx_shift *= (1./np.cos(theta_0)+1.)/2. # fudge factor to reduce slippage more
            #xlamds = np.mean(lamd)
            kx = (Theta - theta_0) * 2. * np.pi / Lambda
            R0H *= np.exp(-1j * kx * dx_shift)
        
        return R0H
    else:
        R00 = np.exp(1j*(chi_0*K0*d/2/gamma_0+A/2*Y1))*(R2-R1)/(R2-R1*np.exp(1j*A/2*(Y1-Y2)));    # forward diffraction amplitude
        #C = np.exp(1j*chi_0*K0*d/(2*gamma_0)); #prompt response
        #R001 = R00-C;
        return R00
    
def Bragg_mirror_reflection(photon_energies_eV, angles_rad, undo_slippageQ=0):
    return Bragg_mirror(photon_energies_eV, angles_rad, reflectionQ=True, undo_slippageQ=undo_slippageQ)
    
def Bragg_mirror_transmission(photon_energies_eV, angles_rad, undo_slippageQ=0):
    return Bragg_mirror(photon_energies_eV, angles_rad, reflectionQ=False, undo_slippageQ=undo_slippageQ)
    
def plot_Bragg_mirror_response_slice(theta_slice=np.pi/4.):
    
    pi = np.pi
    eph = np.linspace(9831.65,9832.25,5001);
    theta_0 = 45.0*pi/180.; dtheta_0 = 50e-6; ntheta_0 = 101;
    #theta_0 *= 0.
    theta = np.linspace(theta_0-dtheta_0,theta_0+dtheta_0,ntheta_0);
    
    R0H = Bragg_mirror_reflection(eph, theta)
    R00 = Bragg_mirror_transmission(eph, theta)

    Eph,Thetaurad = np.meshgrid(eph,1e6*(theta-theta_slice));
    cut = Thetaurad == 0;
    
    plt.plot(Eph[cut],np.abs(R0H[cut])**2)
    plt.title(['Angle = 45 deg'])
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Bragg diffration intensity')
    plt.show()
    
    plt.plot(Eph[cut],np.abs(R00[cut])**2)
    plt.title(['Angle = 45 deg'])
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Forward diffration intensity')
    plt.show()
    
    #plt.plot(Eph[cut],np.abs(C[cut]).^2)
    #plt.title(['Angle = 45 deg'])
    #plt.xlabel('Photon energy (eV)')
    #plt.ylabel('intensity(prompt response)')
    #plt.show()
    
    #plt.plot(Eph[cut],np.abs(R001[cut]).^2)
    #plt.title(['Angle = 45 deg'])
    #plt.xlabel('Photon energy (eV)')
    #plt.ylabel('intensity(forward diffration - prompt response)')
    #plt.show()

# plotting stuff (slow)
def plot_Bragg_mirror_response():
    
    pi = np.pi; ncontours = 100
    eph = np.linspace(9831.65,9832.25,5001);
    theta_0 = 45.0*pi/180.; dtheta_0 = 50e-6; ntheta_0 = 101;
    #theta_0 *= 0.
    theta = np.linspace(theta_0-dtheta_0,theta_0+dtheta_0,ntheta_0);
    
    Eph,Thetaurad = np.meshgrid(eph,1e6*(theta-pi/4));
    
    R0H = Bragg_mirror_reflection(eph, theta)
    R00 = Bragg_mirror_transmission(eph, theta)
    
    plt.contourf(Eph,Thetaurad,np.abs(R0H)**2,ncontours)
    plt.ylabel('Angle - 45 deg (urad)')
    plt.xlabel('Photon energy (eV)')
    plt.title('Bragg diffration intensity')
    plt.colorbar(); plt.show()
    
    plt.contourf(Eph,Thetaurad,np.abs(R00)**2,ncontours)
    plt.ylabel('Angle - 45 deg (urad)')
    plt.xlabel('Photon energy (eV)')
    plt.title('Forward diffration intensity')
    plt.colorbar(); plt.show()
    
    #plt.contourf(Eph,Thetaurad,np.abs(C)**2,ncontours)
    #plt.ylabel('Angle - 45 deg (urad)')
    #plt.xlabel('Photon energy (eV)')
    #plt.title('intensity(prompt response)')
    #plt.colorbar(); plt.show()
    
    #plt.contourf(Eph,Thetaurad,np.abs(R001)**2,ncontours)
    #plt.ylabel('Angle - 45 deg (urad)')
    #plt.xlabel('Photon energy (eV)')
    #plt.title('intensity(forward diffration - prompt response)')
    #plt.colorbar(); plt.show()
    
