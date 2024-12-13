# Condensed-phase-isomerization
(1) First use compile.sh to compile the PO_DVR_2D_GS.cc code.
(2) Unzip the pot_file.zip to extract the 2D potential.
(3) Then run the compiled code : ./PO_DVR_2D_GS &
(4) This code will produce the Eig_n files (containing the eigenfunctions) as well as the vals.dat file containing the eigenvalues.
(5) Identify the "O-down" and "C-down" dominantly localised eigenfunctions and edit the files 12C16O_up_new.py and 12C16O_down_new.py accordingly.
(6) Then run the 12C16O_up_new.py and 12C16O_down_new.py scripts to get the upward and downward transition rates for 1/4 ML 12C16O on NaCl(100).


Note: real_vdos_3CO_nacl.dat contains the vibrational density of states upto the Debye frequency of NaCl , which was computed using ab initio MD simulations at 30 K, for a 1 ML CO/NaCl(100) 
configuration with one "O-down" CO. Here the projected VDOS is saved where the vdos contribution from the "O-down" CO has been subtracted. 
