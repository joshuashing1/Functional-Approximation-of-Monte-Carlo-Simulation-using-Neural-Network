# Personal Proofs

The following proofs are excerpts from question banks of 'Monte Carlo Statistical Methods 2nd Edition' by Christian Robert and George Casella without answer references.

**1.31**

**Part a.** Consider estimation in Linear Model 
![equation](https://latex.codecogs.com/svg.image?Y=b_%7B1%7DX_%7B1%7D&plus;b_%7B2%7DX_%7B2%7D&plus;%5Cepsilon,0%5Cleq%20b_%7B1%7D,b_%7B2%7D%5Cleq%201) for sample ![equation](https://latex.codecogs.com/svg.image?%5Cleft(Y_%7B1%7D,X_%7B11%7D,X_%7B21%7D%5Cright),...,%5Cleft(Y_%7Bn%7D,X_%7B1n%7D,X_%7B2n%7D%5Cright)) where errors i.i.d ![equation](https://latex.codecogs.com/svg.image?%5Cvarepsilon_%7Bi%7D%5Csim%20N(0,1)). 
A noninformative prior is ![equation](https://latex.codecogs.com/svg.image?%5Cpi%5Cleft(b_%7B1%7D,b_%7B2%7D%5Cright)=%5Cmathbb%7BI%7D_%7B%5B0,1%5D%7D%5Cleft(b_%7B1%7D%5Cright)%5Cmathbb%7BI%7D_%7B%5B0,1%5D%7D%5Cleft(b_%7B2%7D%5Cright)). Show that posterior means are given by ![equation](https://latex.codecogs.com/svg.image?i=1,2),
![equation](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5E%7B%5Cpi%7D%5Cleft(b_%7Bi%7D%7Cy_%7B1%7D,...,y_%7Bn%7D%5Cright)=%5Cfrac%7B%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7Db_%7Bi%7D%5Cprod_%7Bj=1%7D%5E%7Bn%7D%5Cvarphi%5Cleft(y_%7Bj%7D-b_%7B1%7DX_%7B1j%7D-b_%7B2%7DX_%7B2j%7Ddb_%7B1%7Ddb_%7B2%7D%5Cright)%7D%7B%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7D%5Cprod_%7Bj=1%7D%5E%7Bn%7D%5Cvarphi%5Cleft(y_%7Bj%7D-b_%7B1%7DX_%7B1j%7D-b_%7B2%7DX_%7B2j%7Ddb_%7B1%7Ddb_%7B2%7D%5Cright)%7D,%5Cvarphi) density standard normal.

*Proof* Given ![equation](https://latex.codecogs.com/svg.image?%5Cvarepsilon_%7Bi%7D%5Csim%20N(0,1),y_%7Bj%7D-b_%7B1j%7Dx_%7B1j%7D-b_%7B2j%7Dx_%7B2j%7D%5Csim%20N(0,1).)
![equation]()
