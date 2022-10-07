Genomic Research Prototypes using Homomorphic Encryption
=====================================

This gitlab repository includes the implementation of Logistic Regression Approximation (LRA) and Chi-Square GWAS protocols described in
"Secure large-scale genome-wide association studies using homomorphic encryption"
by Marcelo Blatt, Alexander Gusev, Yuriy Polyakov, and Shafi Goldwasser.

The repo includes the following files:
* demo-logistic.cpp - research prototype for the LRA protocol.
* demo-chi2.cpp - research prototype for the Chi-Square protocol.
* data/random_sample.csv - an artificial random data set including 3 features, 200 individuals, and 16,384 SNPs (provided solely for demonstration purposes). 

How to Build and Run the Prototypes
=====================================

1. Install PALISADE v1.9.1 from [PALISADE Development Repository](https://gitlab.com/palisade/palisade-development/-/tree/release-v1.9.1). Follow the instructions provided in [README.md](https://gitlab.com/palisade/palisade-development/-/blob/release-v1.9.1/README.md).

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


