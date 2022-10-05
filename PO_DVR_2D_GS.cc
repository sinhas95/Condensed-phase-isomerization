/*******************************************************************************
 * CODE  for solving 2D Hamiltonian for Z and theta for larger grid
 *   with Gram Schmidt reorthogonalization        S. Sinha, F.Bouakline 2022   *   
*******************************************************************************/

#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <armadillo>
#include <random>
#include <cmath>
//#include <omp.h>
#include <iomanip>
//#define NUM_THREADS 20

using namespace arma;
using namespace std;

double normalise(double* A, int N_x){
    double mag = 0.;
    for(int i = 0; i < N_x; i++){
        mag += A[i]*A[i];
    }

    mag = sqrt(mag);
    return mag;
 }


double VVmultiplication(double* vec_a, double* vec_b, int N){
    double sum = 0.;
    int i ;
    //omp_set_num_threads(NUM_THREADS);
    //{ 
        for(i = 0; i < N; i++){
         sum += vec_b[i] * vec_a[i];
        }
    //}
   return sum;
}

int main(int argc, char* argv[]){

 double hartree_cm = 219474.63;
 double eV_hartree = 1./27.211383;
 double a0_ang = 0.52917721;
 int N_p = 1800;
 int N_z = 1400;
 
 double zmax = 11.338401; //bohrradius
 double zmin = 4.3464835; // bohrradius
 double L = 2*M_PI;
 double c_mass = 12.0107 ;
 double o_mass = 15.999;
 double au_mass = 1822.89;
 double M = (c_mass + o_mass)*au_mass;
 double mu = au_mass* ((c_mass*o_mass)/(c_mass + o_mass));
 double R = 2.1567444; // bohrradius C-O interatomic distance
 double I = mu*R*R;
 double dummy;
 
 double dZ = (zmax-zmin)/N_z;
 double dp = L/N_p;
 double L1 = L-dp;
 double L2 = (zmax-dZ)-zmin;
 double Zmin = zmin;
 double Pmin = -L/2.;
 double* P = new double[N_p* N_z];
 double* Z = new double[N_p * N_z];
 double* VPOT = new double[N_p * N_z];
 double vlow = 1000;
 int nb_max = 10000;
 double dq = sqrt(dZ*dp);
 
 arma::mat Hp(N_p, N_p, fill::zeros); // 1D Hamiltonian matrix for theta coordinate
 arma::mat Hz(N_z, N_z, fill::zeros);  // 1D Hamiltonian matrix for Z coordinate
 arma::mat V(N_p, N_z, fill::zeros); // this is where del_V operator is stored
 arma::mat hvec_z_copy(N_z, N_z, fill::zeros);
 arma::mat hvec_p_copy(N_p, N_p, fill::zeros); 
 //Reading the potentials V(Z, theta) and V_ref(theta) V_ref(Z)
 std::ifstream inf; inf.open("e_1400_1800.dat");
 for(int i = 0; i < N_p * N_z; i++){ 
           inf >> P[i] >> Z[i] >> VPOT[i];
           if(VPOT[i]<vlow) vlow = VPOT[i];

 }
 
 for(int i = 0; i < N_p * N_z; i++){ 
      VPOT[i] = VPOT[i] - vlow;
 }
 
 inf.close();
 
 double* ref_Z = new double[N_z];
 double* ref_P = new double[N_p];
 double*  P_2 = new double[N_p];
 double* Z_2 = new double[N_z];
 //calculating del_V with another reference potential V_ref(theta) V_ref(Z)
 std::ofstream outf("Vref_theta.dat");
 for(int i = 0; i < N_p; i++){
     double xpot = 1000;
     double reference_p = 0.;
     for(int j = 0; j < N_z; j++){
         if (VPOT[i*N_z + j] < xpot) {
             xpot = VPOT[i*N_z + j];
             reference_p = P[i*N_z + j];
         }
     }
     ref_P[i] = xpot;
     P_2[i] = reference_p;
     
     outf << P_2[i] << " " << ref_P[i] << "\n";
 }
 
 outf.close();
 
 outf.open("Vref_Z.dat");
 for(int j = 0; j < N_z; j++){
     double zpot = 1000;
     double reference_Z = 0. ;
     for(int i = 0; i < N_p; i++){
         if(VPOT[i*N_z + j] < zpot) {
             zpot = VPOT[i*N_z + j];
             reference_Z = Z[i*N_z +j];
         }
     }
     ref_Z[j] = zpot;
     Z_2[j] = reference_Z;
   //  cout << Z_2[j] << " " << ref_Z[j] << "\n";
     outf << Z_2[j] << " " << ref_Z[j] << "\n";
 }
 outf.close();
 
 for (int i = 0; i < N_p; i++){
     for(int j = 0; j < N_z; j++){
       int p = i*N_z + j; 
       V(i,j) = VPOT[p] - ref_P[i] - ref_Z[j];                
     }
 }

/*------------------------------------------------------------------------------*/ 
 //building up 1D Hamiltonian matrix elements for Z
  //Initialising the Hamiltonian and adding the Vpot.
  for(int i = 0; i < N_z; i++){
      for(int j = 0; j < N_z; j++){
          if(i==j) Hz(i,i) += ref_Z[i];
          else Hz(i, j) = 0.;
      }
  }
  
  
  
  //Evaluating the KEO for Z coordinate using Colbert Miller DVR
 /*  for(int i = 0; i < N_z; i++){
    for(int j = 0; j < N_z; j++){
      if(i==j){ 
	Hz(i,i) += (1./(2.*M*dZ*dZ)) * (pow(M_PI,2))/3.;
      }
      else { 
	Hz(i,j) += (1./(2.*M*dZ*dZ))* (pow(-1,i-j)) * (2./pow((i-j),2));
      }
    }
  }*/ 
  
  //Fourier grid Hamiltonian DVR
  for(int i = 0; i < N_z; i++){
    for(int j = 0; j < N_z; j++){
      if(i==j){ 
	Hz(i,i) += (M_PI*M_PI/(M*L2*L2)*(pow(N_z,2)+2.)/6.) ;
      }
      else { 
	Hz(i,j) += (pow(-1,i-j)*M_PI*M_PI/(M*L2*L2)*1./(pow(sin((i-j)*M_PI/N_z),2))); ;
      }
    }
  }
 
// diagonalising the Hamiltonian matrix Hz of dimension N_z*N_z
// arma::vec Ens_z(N_z, fill::zeros);
// arma::mat hvecs_z(N_z, N_z, fill::zeros);
  arma::vec Ens_z;
  arma::mat hvecs_z;
  eig_sym(Ens_z, hvecs_z, Hz);
 
 outf.open("vals_Z.dat");
 for(int i = 0; i < N_z; i++)
    outf  << setprecision(15) << i << " " << Ens_z(i) << " " << Ens_z(i)*hartree_cm  << "\n";
  outf.close();
 
 //cout << "done0" ;
 
 
/*----------------------------------------------------------------------------*/

//building up Hamiltonian matrix elements for theta
// initialising Hamiltonian and adding Vpot
 for(int i = 0; i < N_p; i++){
      for(int j = 0; j < N_p; j++){
          if(i==j) Hp(i,i) += ref_P[i];
          else Hp(i, j) = 0.;
      }
  }
  
//Evaluating the KEO for theta coordinate
 for(int i = 0; i < N_p; i++){
    for(int j = 0; j < N_p; j++){
      if(i==j){ 
	Hp(i,i) += (M_PI*M_PI/(I*L1*L1)*(pow(N_p,2)+2.)/6.) ;
      }else { 
	Hp(i,j) += (pow(-1,i-j)*M_PI*M_PI/(I*L1*L1)*1./(pow(sin((i-j)*M_PI/N_p),2))) ;
      }
    }
 }

// diagonalising the Hamiltonian matrix Hp of dimension N_p*N_p
 arma::vec Ens_p;
 arma::mat hvecs_p;
 eig_sym(Ens_p, hvecs_p, Hp);
 
 outf.open("vals_theta.dat");
 for(int i = 0; i < N_p; i++)
    outf << setprecision(15) << i << " " << Ens_p(i) << " " << Ens_p(i)*hartree_cm  << "\n";
  outf.close();
 
 /*----------------------------------------------------------------------------*/
//Gram Schmidt reorthogonalization for 1D wavefunctions (theta coordinate)
 //double mag2 = 0.;
 //double norm = 0. ;
 int N_GS_p = 90; //number of wavefunctions to reorthonormalize
 

 double* a_p = new double[N_p];
 double* b_p = new double[N_p];
 double* c_p = new double[N_p];
 double r_ab = 0.;
 
 for(int i = 0; i < N_GS_p; i++){
     for(int k = 0; k < N_p; k++){
         a_p[k] = hvecs_p(k, i);
     }
     for(int j = 0 ; j < i; j++){
         for(int k = 0; k < N_p; k++){
             b_p[k] = hvecs_p(k, j);
         }
         r_ab = VVmultiplication(a_p, b_p, N_p);
         for(int k = 0; k < N_p; k++){
             a_p[k]-= r_ab*b_p[k];
         }
     }
     
     double s = normalise(a_p, N_p);
     for(int l = 0; l < N_p; l++){
         c_p[l] = a_p[l]/s;
         hvecs_p(l, i) = c_p[l];
     }
 }
         
             
 outf.open("vecs_theta.dat");
 for(int i = 0; i < N_p; i++){
     outf << P_2[i] << " " ;
     for(int j = 0; j < 85; j++){
          outf << setprecision(15) << hvecs_p(i, j)/sqrt(dp) << " ";
     }
     outf << "\n";
 }
 outf.close();
//-------------------------------------------------------------------------------
//Gram Schmidt reorthogonalization for 1D wavefunctions (Z coordinate)
 double* a_z = new double[N_z];
 double* b_z = new double[N_z];
 double* c_z = new double[N_z];
 int N_GS_z = 30;
 double r_ab_z = 0.;
 
 for(int i = 0; i < N_GS_z; i++){
     for(int k = 0; k < N_z; k++){
         a_z[k] = hvecs_z(k, i);
     }
     for(int j = 0 ; j < i; j++){
         for(int k = 0; k < N_z; k++){
             b_z[k] = hvecs_z(k, j);
         }
         r_ab_z = VVmultiplication(a_z, b_z, N_z);
         for(int k = 0; k < N_z; k++){
             a_z[k]-= r_ab_z*b_z[k];
         }
     }
     
     double s = normalise(a_z, N_z);
     for(int l = 0; l < N_z; l++){
         c_z[l] = a_z[l]/s;
         hvecs_z(l, i) = c_z[l];
     }
 }
 
 outf.open("vecs_Z.dat");
 for(int i = 0; i < N_z; i++){
     outf << Z_2[i] << " " ;
     for(int j = 0; j < 30; j++){
          outf << setprecision(15) << hvecs_z(i, j)/sqrt(dZ) << " ";
     }
     outf << "\n";
 }
 outf.close();
 
//------------------------------------------------------------------------------------
 //choose zero order product states below a certain energy threshold
 
 outf.open("selected_1D_states.dat");
 
 int count = 0;
 double etot;
 arma::mat ibasis(2, nb_max, fill::zeros);
 for(int i = 0; i < N_p; i++){
     for(int j = 0; j < N_z; j++){
       if(Ens_z(j) <= ref_Z[N_z-1]) {  // selecting 'Z mode' 1D states below the desorption limit ~ 1400 cm^-1
         etot = (Ens_p(i) + Ens_z(j))*hartree_cm;
         if(etot <=4000){ // here 4000 cm^-1 is the energy threshold
             ibasis(0, count) = i;
             ibasis(1, count) = j;
             count += 1 ;
             outf << i << " " << j << " " << etot << "\n";
         }
       }
     }
 }
 cout << "size of final basis" << " " << count << "\n";
 outf.close();
 
 
 //construct full Hamiltonian in the new basis            
 arma::mat H(count,count, fill::zeros);
 
 for(int i = 0; i < count; i++){
     int ip = ibasis(0, i);
     int iz = ibasis(1, i);
     for(int j = 0; j < count; j++){
         int jp = ibasis(0, j);
         int jz = ibasis(1, j);
         
         if(i==j){
             H(i, j) += Ens_p(ip) + Ens_z(iz);
         }
         
         for(int k = 0; k < N_p; k++){
             for(int l = 0; l < N_z; l++){
                 H(i, j) += V(k, l)*hvecs_p(k,ip)*hvecs_p(k,jp)* hvecs_z(l, iz) * hvecs_z(l, jz);
             }
         }
     }
 }
 
 cout << "Construction of full matrix done" << "\n";
 
 //Diagonalising full Hamiltonian matrix
 
 arma::vec Ens;
 arma::mat hvecs;
 eig_sym(Ens, hvecs, H);
 
 //printing eigenvalues
 cout << "Writing down eigenvalues" << "\n";
 outf.open("vals.dat");
 for( int i = 0; i < count; i++)
    outf << setprecision(15) << i << " " << Ens(i) << " " << Ens(i)*hartree_cm  << "\n";
 outf.close();
 
 cout << "Writing down eigenvectors" << "\n";
 arma::mat vect(N_p*N_z, count, fill::zeros);
 //converting eigenvectors into old basis
 for(int l = 0; l < count; l++){
     for(int i = 0; i < N_p; i++){
         for(int j = 0; j < N_z; j++){
            // outf << P[i*N_z+j] <<  " " << Z[i*N_z+j] << " ";
             //double vect_val = 0.;
             for(int m = 0; m < count; m++){                 
                int jp = ibasis(0, m);
                int jz = ibasis(1, m);
                vect(i*N_z + j, l) += hvecs(m, l)* hvecs_p(i, jp)* hvecs_z(j, jz);
             }
             
         }
     }
 }
 //----------------------------------------------------------------------------------------------------------------------------
 //doing Gram Schmidt orthogonalisation 
 int N_GS_pz = 350; //number of wavefunctions to reorthonormalize
 

 double* a_pz = new double[N_p*N_z];
 double* b_pz = new double[N_p*N_z];
 double* c_pz = new double[N_p*N_z];
 double r_ab_pz = 0.;
 
 for(int i = 0; i < N_GS_pz; i++){
     for(int k = 0; k < N_p*N_z; k++){
         a_pz[k] = vect(k, i);
     }
     for(int j = 0 ; j < i; j++){
         for(int k = 0; k < N_p*N_z; k++){
             b_pz[k] = vect(k, j);
         }
         r_ab_pz = VVmultiplication(a_pz, b_pz, N_p*N_z);
         for(int k = 0; k < N_p*N_z; k++){
             a_pz[k] -= r_ab_pz*b_pz[k];
         }
     }
     
     double spz = normalise(a_pz, N_p*N_z);
     for(int l = 0; l < N_p*N_z; l++){
         c_pz[l] = a_pz[l]/spz;
         vect(l, i) = c_pz[l];
     }
 }
 //printing eigenvectors
 std::ofstream outf2;
 for( int l = 0; l < 350; l++){
   char prhofile[1024];
   snprintf(prhofile, sizeof(prhofile), "Eig_%d.dat", l);
   outf2.open(prhofile);
   for(int i = 0; i < N_p; i++){
     for(int j = 0; j < N_z; j++){
       outf2  << setprecision(15) <<  P[i*N_z+j] <<  " " << Z[i*N_z+j] << " " << vect(i*N_z+j, l)/dq << "\n";
     }
  
     outf2 << "\n";
   }
   outf2.close();
    
  }    
         
}//end of program
 
