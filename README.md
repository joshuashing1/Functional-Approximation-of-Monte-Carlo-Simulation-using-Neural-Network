# Personal Proofs

The following proofs are excerpts from question banks of 'Monte Carlo Statistical Methods 2nd Edition' by Christian Robert and George Casella without answer references.

**1.31 Part a.**
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BConsider%20estimation%20in%20Linear%20Model%5C;%7DY=b_%7B1%7DX_%7B1%7D&plus;b_%7B2%7DX_%7B2%7D&plus;%5Cvarepsilon,)
![equation](https://latex.codecogs.com/svg.image?%5C:0%5Cleq%20b_%7B1%7D,b_%7B2%7D%5Cleq1%5Ctext%7B%5C:for%20samples%5C:%7D(Y_%7B1%7D,X_%7B11%7D,X_%7B21%7D),...,(Y_%7Bn%7D,X_%7B1n%7D,X_%7B2n%7D).)
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7B%5C:Errors%20are%20i.i.d%5C:%7D%5Cvarepsilon_%7Bi%7D%5Csim%20N(0,1).%5Ctext%7B%5C:A%20noninformative%20prior%20is%5C:%7D%5Cpi(b_%7B1%7D,b_%7B2%7D)=%5Cmathbb%7BI%7D_%7B%5B0,1%5D%7D(b_%7B1%7D)%5Cmathbb%7BI%7D_%7B%5B0,1%5D%7D(b_%7B2%7D).%5Ctext%7B%5C:Show%20that%20posterior%20means%20are%20given%20by%5C:%7D)
![equation](https://latex.codecogs.com/svg.image?i=1,2,%5Cmathbb%7BE%5E%7B%5Cpi%7D%7D(b_%7Bi%7D%7Cy_%7B1%7D,...,y_%7Bn%7D)=%5Cfrac%7B%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7Db_%7Bi%7D%5Cprod_%7Bj=1%7D%5E%7Bn%7D%5Cvarphi(y_%7Bj%7D-b_%7B1%7DX_%7B1j%7D-b_%7B2%7DX_%7B2j%7D)db_%7B1%7Ddb_%7B2%7D%7D%7B%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7D%5Cprod_%7Bj=1%7D%5E%7Bn%7D%5Cvarphi(y_%7Bj%7D-b_%7B1%7DX_%7B1j%7D-b_%7B2%7DX_%7B2j%7D)db_%7B1%7Ddb_%7B2%7D%7D,%5Cvarphi%5Ctext%7B%5C:is%20density%20standard%20model.)

*Proof.* 
![equation](https://latex.codecogs.com/svg.image?%5Ctext%7BGiven%5C:%7D%5Cvarepsilon_%7Bi%7D%5Csim%20N(0,1),%5C:y_%7Bj%7D-b_%7B1j%7Dx_%7B1j%7D-b_%7B2j%7Dx_%7B2j%7D%5Csim%20N(0,1).%5C:%5Cmathbb%7BE%5E%7B%5Cpi%7D%7D%5Cleft(b_%7Bi%7D%7Cy_%7B1%7D,...,y_%7Bn%7D%5Cright)=%5Cint%20b_%7Bi%7D%5Cpi(b_%7B1%7D,b_%7B2%7D%7C(x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D))db_%7B1%7Ddb_%7B2%7D.%5Ctext%7B%5C:Since%5C:%7D%5Cpi(b_%7B1%7D,b_%7B2%7D%7C(x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D))%5Ctext%7B%5C;directly%20proportional%20to%5C;%7D%5Cpi(b_%7B1%7D,b_%7B2%7D)f((x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D)%7Cb_%7B1%7D,b_%7B2%7D),%5Ctext%7B%5C:WLOG%5C:%7D%5Cint%20b_%7Bi%7D%5Cpi(b_%7B1%7D,b_%7B2%7D%7C(x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D))db_%7B1%7Ddb_%7B2%7D=%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7Db_%7Bi%7Dk%5Cpi(b_%7B1%7D,b_%7B2%7D)f((x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D)%7Cb_%7B1%7D,b_%7B2%7D)db_%7B1%7Ddb_%7B2%7D=k%5Cint_%7B0%7D%5E%7B1%7D%5Cint_%7B0%7D%5E%7B1%7Db_%7Bi%7D%5Cpi(b_%7B1%7D,b_%7B2%7D)f((x_%7B11%7D,x_%7B21%7D),...,(x_%7B1n%7D,x_%7B2n%7D)%7Cb_%7B1%7D,b_%7B2%7D)db_%7B1%7Ddb_%7B2%7D=)
![equation]
![equation]
![equation]
![equation]
![equation]
![equation]
![equation]
![equation]
