Genomic Research Prototypes using Homomorphic Encryption
=====================================

This gitlab repository includes the implementation of Logistic Regression Approximation (LRA) and Chi-Square GWAS protocols described in
[Secure large-scale genome-wide association studies using homomorphic encryption](https://www.pnas.org/doi/full/10.1073/pnas.1918257117)
by Marcelo Blatt, Alexander Gusev, Yuriy Polyakov, and Shafi Goldwasser.

This code was originally written using PALISADE (https://gitlab.com/duality-technologies-public/palisade-gwas-demos/) and then migrated over to OpenFHE.

The repo includes the following files:
* demo-logistic.cpp - research prototype for the LRA protocol.
* demo-chi2.cpp - research prototype for the Chi-Square protocol.
* data/random_sample.csv - an artificial random data set including 3 features, 200 individuals, and 16,384 SNPs (provided solely for demonstration purposes). 

How to Build and Run the Prototypes
=====================================

1. Install OpenFHE from [OpenFHE Development Repository](https://github.com/openfheorg/openfhe-development). Follow the instructions provided in [README.md](https://github.com/openfheorg/openfhe-genomic-examples/blob/main/README.md).

2. Clone this repository to a local directory and switch to this directory. 

3. Create a directory where the binaries will be built. The typical choice is a subfolder "build". In this case, run the following commands:
```
mkdir build
cd build
cmake ..
make
```
4. Run the following command to execute the LRA prototype (change sample size and number of SNPs as needed):
```
./demo-logistic --SNPdir "../data" --SNPfilename "random_sample" --pvalue "pvalue.txt" --runtime "result.txt" --samplesize="200" --snps="16384"
```

or

Run the following command to execute the Chi-Square prototype (change sample size and number of SNPs as needed):
```
./demo-chi2 --SNPdir "../data" --SNPfilename "random_sample" --pvalue "pvalue.txt" --runtime "result.txt" --samplesize="200" --snps="16384"
```

5. The results will be written to the "data" folder. The following output files will be created for both prototypes:

* pvalue.txt - p-values for each SNP
* result.txt - runtime metrics

Additional files with outputs of protocol-specific statistics will also be created.


