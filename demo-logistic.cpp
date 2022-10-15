/***
 * Copyright (c) 2020-2022 Duality Technologies, Inc.
 * Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
 * See the LICENSE.md file for the full text of the license.
 * If you share the Licensed Material (including in modified form) you must include the above attribution in the copy you share.
 ***/
/*

Implementation for the Logistic Regression Approximation GWAS solution described in
"Secure large-scale genome-wide association studies using homomorphic encryption"
by Marcelo Blatt, Alexander Gusev, Yuriy Polyakov, and Shafi Goldwasser

Command to execute the prototype
./demo-logistic --SNPdir "../data" --SNPfilename "random_sample" --pvalue "pvalue.txt" --runtime "result.txt" --samplesize="200" --snps="16384"

*/

#include <getopt.h>
#include <numeric>
#include <cmath>

#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

void RunLogReg(const string &SNPDir, const string &SNPFileName, const string &pValue,
		const string &Runtime, const string &SampleSize, const string &SNPs);

Ciphertext<DCRTPoly> zExpand(const Ciphertext<DCRTPoly> p, const Ciphertext<DCRTPoly> y);

shared_ptr<std::vector<std::vector<Ciphertext<DCRTPoly>>>> MatrixInverse(const Ciphertext<DCRTPoly> m, size_t k, CiphertextImpl<DCRTPoly> &b, CiphertextImpl<DCRTPoly> &cd,
		const std::map<usint, EvalKey<DCRTPoly>> &map, const std::map<usint, EvalKey<DCRTPoly>> &rotKeys,
		const std::map<usint, EvalKey<DCRTPoly>> &evalSumRows);

Ciphertext<DCRTPoly> CloneCiphertext(const Ciphertext<DCRTPoly> ciphertext, size_t size,
		const std::map<usint,EvalKey<DCRTPoly>> &rotKeys, const std::map<usint,EvalKey<DCRTPoly>> &evalSumRows);

shared_ptr<std::vector<Ciphertext<DCRTPoly>>> SplitIntoSingle(const Ciphertext<DCRTPoly> c, size_t N, size_t k,
		const std::map<usint, EvalKey<DCRTPoly>> &rotKeys);

Ciphertext<DCRTPoly> BinaryTreeAdd(std::vector<Ciphertext<DCRTPoly>> &vector);

void CompressEvalKeys(std::map<usint, EvalKey<DCRTPoly>> &ek, size_t level);

void ReadSNPFile(vector<string>& headers, std::vector<std::vector<double>> & dataColumns,std::vector<std::vector<double>> &x, std::vector<double> &y,
		string dataFileName, size_t N, size_t M);

Ciphertext<DCRTPoly> HoistedAutomorphism(const EvalKey<DCRTPoly> ek,
		ConstCiphertext<DCRTPoly> cipherText, const shared_ptr<vector<DCRTPoly>> digits, const usint index);

double normalCFD(double value) { return 0.5 * erfc(-value * M_SQRT1_2); }

double sf(double value) { return 1 - normalCFD(value); }

double BS(double z) {
	double y = exp(-z*z/2);
	return sqrt(1-y) * (31*y/200 - 341*y*y/8000) / sqrt(M_PI);
}

int main(int argc, char **argv) {

	int opt;

	static struct option long_options[] =
	  {
		/* These options dont set a flag.
		   We distinguish them by their indices. */
		{"SNPdir",  	required_argument, 			0, 'S'},
		{"SNPfilename",  	required_argument, 			0, 's'},
		{"pvalue",  	required_argument, 			0, 'p'},
		{"runtime",  	required_argument, 			0, 'r'},
		{"samplesize",  	required_argument, 			0, 'N'},
		{"snps",  	required_argument, 			0, 'M'},
		{0, 0, 0, 0}
	  };

	/* getopt_long stores the option index here. */
	int option_index = 0;

	string SNPDir;
	string SNPFileName;
	string pValue;
	string Runtime;
	string SampleSize;
	string SNPs;

	while ((opt = getopt_long(argc, argv, "S:s:p:r:N:M", long_options, &option_index)) != -1) {
		switch (opt)
		{
			case 'S':
				SNPDir = optarg;
				break;
			case 's':
				SNPFileName = optarg;
				break;
			case 'p':
				pValue = optarg;
				break;
			case 'r':
				Runtime = optarg;
				break;
			case 'N':
				SampleSize = optarg;
				break;
			case 'M':
				SNPs = optarg;
				break;
			default: /* '?' */
			  std::cerr<< "Usage: "<<argv[0]<<" <arguments> " <<std::endl
				   << "arguments:" <<std::endl
				   << "  -S --SNPdir SNP file directory"  <<std::endl
				   << "  -s --SNPfilename SNP file name"  <<std::endl
				   << "  -o --pvalue p-values file"  <<std::endl
				   << "  -r --runtime runtime output file name"  <<std::endl
				   << "  -N --samplesize number of individuals"  <<std::endl
			   	   << "  -M --snps number of SNPs"  <<std::endl;
			  exit(EXIT_FAILURE);
		}
	}

	RunLogReg(SNPDir, SNPFileName, pValue, Runtime, SampleSize, SNPs);

	return 0;

}

void RunLogReg(const string &SNPDir, const string &SNPFileName, const string &pValue, const string &Runtime, const string &SampleSize, const string &SNPs) {

	TimeVar t;
	TimeVar tAll;

	TIC(tAll);

	double keyGenTime(0.0);
	double encryptionTime(0.0);
	double computation1Time(0.0);
	double computation2Time(0.0);
	double computation3Time(0.0);
	double computation4Time(0.0);
	double computation5Time(0.0);
	double computation6Time(0.0);
	double computation7Time(0.0);
	double computation8Time(0.0);
	double computationTime(0.0);
	double decryptionTime(0.0);
	double endToEndTime(0.0);

	std::cout << "\n======LOGISTIC REGRESSION SOLUTION========\n" << std::endl;

	vector<string> headers1;
	vector<string> headersS;

	std::vector<std::vector<double>> xData;
	std::vector<double> yData;
	std::vector<std::vector<double>> sData;

	size_t N = std::stoi(SampleSize);
	size_t M = std::stoi(SNPs);

	double scalingFactor = 1e-1;
	double scalingFactorD = 1e-2;
	double scalingFactorN = 1e-2;

	ReadSNPFile(headersS,sData,xData,yData,SNPDir + "/" + SNPFileName,N,M);

	N = sData.size();
	M = sData[0].size();
	size_t k =  xData[0].size();

	usint m;

	m = 65536;

	size_t n = m/4;

	usint init_size = 17;
	usint dcrtBits = 50;

	size_t k2 = k*k;

	usint batchSize = k*k;

	CCParams<CryptoContextCKKSRNS> parameters;

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetMultiplicativeDepth(init_size-1);
	parameters.SetRingDim(m/2);
	parameters.SetScalingTechnique(FIXEDMANUAL);
	parameters.SetKeySwitchTechnique(BV);
	parameters.SetFirstModSize(dcrtBits);
	parameters.SetBatchSize(k);
	parameters.SetMaxRelinSkDeg(4);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);

	std::cout << "\nNumber of Individuals = " << N << std::endl;
	std::cout << "Number of SNPs = " << M << std::endl;
	std::cerr << "Number of features = " <<  k << std::endl;

	TIC(t);

	auto keyPair = cc->KeyGen();
	cc->EvalMultKeysGen(keyPair.secretKey);
	cc->EvalSumKeyGen(keyPair.secretKey);
	auto evalSumRows = cc->EvalSumRowsKeyGen(keyPair.secretKey, nullptr, k);
	auto evalSumCols = cc->EvalSumColsKeyGen(keyPair.secretKey, nullptr);
	// EvalSum key is also used for rotations by 1 and 2
	auto evalSum = cc->GetEvalSumKeyMap(keyPair.secretKey->GetKeyTag());

	auto pubKeyS = PublicKey<DCRTPoly>(new PublicKeyImpl<DCRTPoly>(*keyPair.publicKey));
	std::vector<DCRTPoly> pubElementsS = pubKeyS->GetPublicElements();
	for (size_t i=0; i < pubElementsS.size(); i++)
		pubElementsS[i].DropLastElements(10);
	pubKeyS->SetPublicElements(pubElementsS);

	auto pubKeyX = PublicKey<DCRTPoly>(new PublicKeyImpl<DCRTPoly>(*keyPair.publicKey));
	auto pubElementsX = pubKeyX->GetPublicElements();
	for (size_t i=0; i < pubElementsX.size(); i++)
		pubElementsX[i].DropLastElements(11);
	pubKeyX->SetPublicElements(pubElementsX);

	std::vector<int32_t> indicesM;
	for (size_t i = 3; i < k*k; i++) {
		if (!((i == 4) || (i == 8)))
			indicesM.push_back(i);
	}

	cc->SetKeyGenLevel(5);

	auto rotKeysM = cc->GetScheme()->EvalAtIndexKeyGen(nullptr,keyPair.secretKey, indicesM);

	std::vector<int32_t> indicesConv;
	for (size_t i = 4; i < m/4; i=2*i)
		indicesConv.push_back(m/4-i);

	cc->SetKeyGenLevel(8);

	auto rotKeysConv = cc->GetScheme()->EvalAtIndexKeyGen(nullptr,keyPair.secretKey, indicesConv);

	keyGenTime = TOC(t);

	TIC(t);

	uint32_t numCt = (uint32_t)std::ceil((double)(N*k*k)/(double)(n));

	std::cerr << "Number of ciphertexts: " << numCt << std::endl;

	size_t Nfull = n/k2;

	std::vector<std::vector<std::complex<double>>> x;
	for (size_t r = 0; r < numCt; r++) {
		std::vector<std::complex<double>> xTemp;
		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		for (size_t j = 0; j < N1; j++)
			for (size_t i = 0; i < k; i++)
				for (size_t p = 0; p < k; p++)
					xTemp.push_back(xData[j+r*Nfull][p]);
		x.push_back(xTemp);
	}

	std::vector<std::vector<std::complex<double>>> xt;
	for (size_t r = 0; r < numCt; r++) {
		std::vector<std::complex<double>> xTemp;
		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		for (size_t j = 0; j < N1; j++)
			for (size_t i = 0; i < k; i++)
				for (size_t p = 0; p < k; p++)
					xTemp.push_back(xData[j+r*Nfull][i]);
		xt.push_back(xTemp);
	}

	std::vector<std::vector<std::complex<double>>> y;
	for (size_t r = 0; r < numCt; r++) {
		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		std::vector<std::complex<double>> yTemp;
		for (size_t j = 0; j < N1; j++)
			for (size_t i = 0; i < k*k; i++) {
			{
				yTemp.push_back(yData[j+r*Nfull]);
			}
		}
		y.push_back(yTemp);
	}

	vector<Plaintext> X(numCt);
	vector<Plaintext> XT(numCt);
	vector<Plaintext> Y(numCt);
	for (size_t r = 0; r < numCt; r++) {
		X[r] = cc->MakeCKKSPackedPlaintext(x[r],1,0,nullptr,m/4);
		XT[r] = cc->MakeCKKSPackedPlaintext(xt[r],1,0,nullptr,m/4);
		Y[r] = cc->MakeCKKSPackedPlaintext(y[r],1,0,nullptr,m/4);
	}

	std::vector<vector<Ciphertext<DCRTPoly>>> cX1(N);

	size_t sizeS = (size_t)std::ceil((double)sData[0].size()/(m/4));

	std::vector<std::vector<std::vector<std::complex<double>>>> sDataArray(sizeS);

	for(size_t s = 0; s < sizeS; s++)
		sDataArray[s] = std::vector<std::vector<std::complex<double>>>(sData.size());

	for (size_t i=0; i < sData.size(); i++){

		for(size_t s = 0; s < sizeS; s++)
			sDataArray[s][i] = std::vector<std::complex<double>>(m/4);

		size_t counter = 0;

		for (size_t j=0; j<sData[i].size(); j++) {
			if ((j>0) && (j%(m/4)==0))
				counter++;
			sDataArray[counter][i][j%(m/4)] = scalingFactor*sData[i][j];
		}
	}

	std::vector<std::vector<Ciphertext<DCRTPoly>>> S(sizeS);

	for (size_t i = 0; i < sizeS; i++)
		S[i] = std::vector<Ciphertext<DCRTPoly>>(N);

	//Encryption of single-integer ciphertexts
#pragma omp parallel for
	for (size_t i=0; i<N; i++){
		for (size_t s=0; s < sizeS; s++){
			Plaintext sTemp = cc->MakeCKKSPackedPlaintext(sDataArray[s][i],1,10,pubElementsS[0].GetParams(),m/4);
			S[s][i] = cc->Encrypt(pubKeyS, sTemp);
		}
		std::vector<Ciphertext<DCRTPoly>> x1Temp;
		for (size_t j=0; j<k; j++){
			std::vector<std::complex<double>> xVector(m/4,xData[i][j]);
			Plaintext xTemp = cc->MakeCKKSPackedPlaintext(xVector,1,11,pubElementsX[0].GetParams(),m/4);
			x1Temp.push_back(cc->Encrypt(pubKeyX, xTemp));
		}
		cX1[i] = x1Temp;
	}

	vector<Ciphertext<DCRTPoly>> cX(numCt);
	vector<Ciphertext<DCRTPoly>> cXT(numCt);
	vector<Ciphertext<DCRTPoly>> cY(numCt);

	for (size_t r=0; r<numCt; r++){
		cX[r] = cc->Encrypt(keyPair.publicKey, X[r]);
		cXT[r] = cc->Encrypt(keyPair.publicKey, XT[r]);
		cY[r] = cc->Encrypt(keyPair.publicKey, Y[r]);
	}

	encryptionTime = TOC(t);

	TIC(t);

	double alpha = 0.00358;

	// Compute p1 = X^T (y - 0.5)
	Ciphertext<DCRTPoly> cP1Sum, cP1;
	for (size_t r=0; r<numCt; r++){
		cP1 = cc->EvalMult(cX[r],cc->EvalSub(cc->EvalSub(cY[r],double(0.25)),double(0.25)));
		cP1 = cc->EvalSumRows(cP1,k*k,*evalSumRows);
		//cP1 = cc->ModReduce(cP1);
		if (r==0)
			cP1Sum = cP1;
		else
			cP1Sum = cc->EvalAdd(cP1Sum,cP1);
	}
	cP1Sum = cc->ModReduce(cP1Sum);

	// Compute p2 = 0.15625*alpha*X
	vector<Ciphertext<DCRTPoly>> cP2Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cP2 = cc->EvalMult(cX[r],double(0.15625*alpha));
		cP2Arr[r] = cc->ModReduce(cP2);
	}

	//Compute p3 = p1*p2
	vector<Ciphertext<DCRTPoly>> cP3Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cP3 = cc->EvalMult(cP1Sum,cP2Arr[r]);
		cP3 = cc->EvalSumCols(cP3,k,*evalSumCols);
		cP3 = cc->ModReduce(cP3);
		cP3Arr[r] = cc->ModReduce(cP3);
	}

	//Compute p = p3 + 0.5
	vector<Ciphertext<DCRTPoly>> cPArr(numCt);
	for (size_t r=0; r<numCt; r++){
		//auto cP = cc->EvalSub(cP3Arr[r],cP6Arr[r]);
		auto cP = cc->EvalAdd(cP3Arr[r],double(0.25));
		cPArr[r] = cc->EvalAdd(cP,double(0.25));
	}

	// Compute p^2
	vector<Ciphertext<DCRTPoly>> cPSquareArr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cPSquare = cc->EvalMult(cPArr[r],cPArr[r]);
		cPSquareArr[r] = cc->ModReduce(cPSquare);
	}

	// Compute w = p - p^2
	vector<Ciphertext<DCRTPoly>> cWArr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cPReduced = cc->LevelReduce(cPArr[r],nullptr);
		cWArr[r] = cc->EvalSub(cPReduced,cPSquareArr[r]);
	}

	// Computes zExpand
	vector<Ciphertext<DCRTPoly>> cZArr(numCt);
	for (size_t r=0; r<numCt; r++){
		cZArr[r] = zExpand(cPArr[r], cY[r]);
	}

	// Compute x^T diag(w)
	vector<Ciphertext<DCRTPoly>> cM1Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cXTNReduced = cc->LevelReduce(cXT[r],nullptr,4);
		auto cM1 = cc->EvalMult(cXTNReduced,cWArr[r]);
		cM1Arr[r] = cc->ModReduce(cM1); // Level 5
	}

	CompressEvalKeys(*evalSumRows,5);

	//Compute M = (x^T diag(w)) X
	Ciphertext<DCRTPoly> cMSum, cM;
	vector<Ciphertext<DCRTPoly>> cXReducedArr(numCt);
	for (size_t r=0; r<numCt; r++){
		cXReducedArr[r] = cc->LevelReduce(cX[r],nullptr,5); //Level 5
		cM = cc->EvalMult(cM1Arr[r],cXReducedArr[r]);
		cM = cc->EvalSumRows(cM,k*k,*evalSumRows);
		if (r==0)
			cMSum = cM;
		else
			cMSum = cc->EvalAdd(cMSum,cM);
	}

	computation1Time = TOC(t);

	TIC(t);

	Ciphertext<DCRTPoly> cB(new CiphertextImpl<DCRTPoly>(cc));
	Ciphertext<DCRTPoly> cd(new CiphertextImpl<DCRTPoly>(cc));

	CompressEvalKeys(evalSum,5);

	auto cB1 = MatrixInverse(cMSum,k,*cB,*cd, evalSum, *rotKeysM, *evalSumRows);

	computation2Time = TOC(t);

	TIC(t);

	cMSum = cc->ModReduce(cMSum);

	for (size_t r=0; r<numCt; r++){
		cM1Arr[r] = cc->LevelReduce(cM1Arr[r],nullptr,2); //Level 7
	}

	// Compute (X^T diag(w)) z
	Ciphertext<DCRTPoly> cztr1Sum, cztr1;
	for (size_t r=0; r<numCt; r++){
		cztr1 = cc->EvalMult(cM1Arr[r],cZArr[r]);
		cztr1 = cc->EvalSumRows(cztr1,k*k,*evalSumRows);
		//cztr1 = cc->ModReduce(cztr1); // Level 8
		if (r==0)
			cztr1Sum = cztr1;
		else
			cztr1Sum = cc->EvalAdd(cztr1Sum,cztr1);
	}
	cztr1Sum = cc->ModReduce(cztr1Sum);

	// Compute XB
	vector<Ciphertext<DCRTPoly>> cztr2Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		cXReducedArr[r] = cc->LevelReduce(cXReducedArr[r],nullptr,4); //Level 9
		auto cztr2 = cc->EvalMult(cXReducedArr[r],cB);
		cztr2 = cc->EvalSumCols(cztr2,k,*evalSumCols);
		cztr2 = cc->ModReduce(cztr2);
		cztr2Arr[r] = cc->ModReduce(cztr2); //Level 11
	}

	CompressEvalKeys(*rotKeysM,6);
	CompressEvalKeys(*evalSumRows,6);

	// Compute the product of (XB) and (X^T diag(w)) z
	vector<Ciphertext<DCRTPoly>> cztr3Arr(numCt);
	cztr1Sum = cc->LevelReduce(cztr1Sum,nullptr,3); //Level 11
	for (size_t r=0; r<numCt; r++){

		auto cztr3 = cc->EvalMult(cztr1Sum,cztr2Arr[r]);
		auto cztr4 = cc->EvalAdd(cztr3,cc->GetScheme()->EvalAtIndex(cztr3,4,*evalSumRows));
		auto cztr5 = cc->EvalAdd(cc->GetScheme()->EvalAtIndex(cztr3,8,*evalSumRows),cc->GetScheme()->EvalAtIndex(cztr3,12,*rotKeysM));
		auto cztr6 = cc->EvalAdd(cztr4,cztr5);
		cztr6 = cc->ModReduce(cztr6); //Level 12
		std::vector<std::complex<double>> mask(m/4);

		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;

		size_t NPow2cur = 1<<(size_t)std::ceil(log2(N1));

		for (size_t i = 0; i < NPow2cur*k*k; i++)
		{
			if ((i % batchSize == 0) || (i % batchSize == 1) || (i % batchSize == 2) || (i % batchSize == 3))
				mask[i] = 1;
			else
				mask[i] = 0;
		}
		Plaintext plaintextMask = cc->MakeCKKSPackedPlaintext(mask,1,0,nullptr,m/4);
		auto cMask = cc->EvalMult(cztr6,plaintextMask);
		cztr4 = cc->EvalAdd(cMask,cc->GetScheme()->EvalAtIndex(cMask,m/4-4,*rotKeysConv));
		cztr3Arr[r] = cc->EvalAdd(cztr4,cc->GetScheme()->EvalAtIndex(cztr4,m/4-8,*rotKeysConv));

	}

	rotKeysM->clear();
	evalSumRows->clear();

	//Compute d*z
	auto cdDen = cc->LevelReduce(cd,nullptr,1); //Level 10
	cd = cc->LevelReduce(cd,nullptr,3); //Level 12
	vector<Ciphertext<DCRTPoly>> cztr4Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		cZArr[r] = cc->LevelReduce(cZArr[r],nullptr,5); //Level 12
		cztr4Arr[r] = cc->EvalMult(cZArr[r],cd);
	}

	//Computer ztr
	vector<Ciphertext<DCRTPoly>> cztrArr(numCt);
	for (size_t r=0; r<numCt; r++){
		auto cztr = cc->EvalSub(cztr4Arr[r],cztr3Arr[r]);
		cztrArr[r] = cc->ModReduce(cztr); //Level 13
	}

	computation3Time = TOC(t);

	TIC(t);

	vector<Ciphertext<DCRTPoly>> cWConvArr(numCt);
	for (size_t r=0; r<numCt; r++){

		cWArr[r] = cc->LevelReduce(cWArr[r],nullptr,4); //Level 8

		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		size_t NPow2cur = 1<<(size_t)std::ceil(log2(N1));

		std::vector<std::complex<double>> maskW(m/4);
		for (size_t v = 0; v < NPow2cur*k*k; v++)
			maskW[v] = 1;
		for (size_t v = NPow2cur*k*k; v < m/4; v++)
			maskW[v] = 0;
		Plaintext plaintextW = cc->MakeCKKSPackedPlaintext(maskW,1,0,nullptr,m/4);
		auto cWConv = cc->EvalMult(cWArr[r],plaintextW);

		for (size_t j = NPow2cur*k*k; j < m/4; j=j*2 ) {
			cWConv = cc->EvalAdd(cWConv,cc->GetScheme()->EvalAtIndex(cWConv,m/4-j,*rotKeysConv));
		}

		cWConvArr[r] = cc->ModReduce(cWConv); //Level 9

	}

	CompressEvalKeys(*rotKeysConv,1);

	vector<Ciphertext<DCRTPoly>> cWVector;
	for (size_t r=0; r<numCt; r++){
		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		auto temp = SplitIntoSingle(cWConvArr[r], N1, k, *rotKeysConv); //Level 10
		cWVector.insert(cWVector.end(),temp->begin(),temp->end()); //Level 10
		cWArr[r] = cc->LevelReduce(cWArr[r],nullptr,1); //Level 9
	}

	computation4Time = TOC(t);

	TIC(t);

	std::vector<std::vector<Ciphertext<DCRTPoly>>> strVector1(sizeS);

	for (size_t i = 0; i < sizeS; i++)
		strVector1[i] = std::vector<Ciphertext<DCRTPoly>>(N);

	// Compute diag(w) S
	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for (size_t i = 0; i < N; i++)
		{
			strVector1[s][i] = cc->EvalMultNoRelin(cWVector[i],S[s][i]);
			strVector1[s][i] = cc->ModReduce(strVector1[s][i]); // Level 11
		}
	}

	// Compute X^T (diag(w) S)
	std::vector<std::vector<Ciphertext<DCRTPoly>>> strVector2(sizeS);

	for (size_t s = 0; s < sizeS; s++) {
		strVector2[s] = std::vector<Ciphertext<DCRTPoly>>(k);
	}

	for (size_t j = 0; j < k; j++)
	{
		for (size_t s = 0; s < sizeS; s++) {
			std::vector<Ciphertext<DCRTPoly>> tempVector(N);
#pragma omp parallel for
			for (size_t i = 0; i < N; i++)
			{
					tempVector[i] = cc->EvalMultNoRelin(strVector1[s][i],cX1[i][j]);

			}
			//std::cerr << "passed mult" << std::endl;
			strVector2[s][j] = BinaryTreeAdd(tempVector);
			tempVector.clear();
			//std::cerr << "passed binary tree" << std::endl;
			strVector2[s][j] = cc->ModReduce(strVector2[s][j]); //Level 12
		}
	}

	// Compute B X^T (diag(w) S)
	std::vector<std::vector<Ciphertext<DCRTPoly>>> strVector3(sizeS);

	for (size_t s = 0; s < sizeS; s++)
		strVector3[s] = std::vector<Ciphertext<DCRTPoly>>(k);

#pragma omp parallel for
	for(size_t i = 0; i < k; i++) {
		(*cB1)[i][0] = cc->LevelReduce((*cB1)[i][0],nullptr,3);
			for(size_t j=1; j < k; j++) {
				(*cB1)[i][j] = cc->LevelReduce((*cB1)[i][j],nullptr,3);
			}
	}


	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for(size_t i = 0; i < k; i++) {
			auto temp = cc->EvalMultAndRelinearize((*cB1)[i][0],strVector2[s][0]);
			for(size_t j=1; j < k; j++) {
				temp = cc->EvalAdd(temp,cc->EvalMultAndRelinearize((*cB1)[i][j],strVector2[s][j]));
			}
			temp = cc->ModReduce(temp); //Level 13
			strVector3[s][i] = temp;
		}
	}

#pragma omp parallel for
	for(size_t i = 0; i < N; i++) {
		cX1[i][0] = cc->LevelReduce(cX1[i][0],nullptr,2);
		for(size_t j=1; j < k; j++) {
			cX1[i][j] = cc->LevelReduce(cX1[i][j],nullptr,2);
		}
	}

	// Compute X B X^T (diag(w) S)
	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for(size_t i = 0; i < N; i++) {
			auto temp = cc->EvalMultNoRelin(cX1[i][0],strVector3[s][0]);
			for(size_t j=1; j < k; j++) {
				temp = cc->EvalAdd(temp,cc->EvalMultNoRelin(cX1[i][j],strVector3[s][j]));
			}
			strVector1[s][i] = temp;
		}
	}

	// Compute d S - X B X^T (diag(w) S)
	//auto cd12 = cd;
	cd = cc->LevelReduce(cd,nullptr,1); //Level 13
	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for (size_t i = 0; i < N; i++)
		{
			S[s][i] = cc->LevelReduce(S[s][i],nullptr,3);
			strVector1[s][i] = cc->EvalSub(cc->EvalMultNoRelin(cd,S[s][i]),strVector1[s][i]); // Level 13
			strVector1[s][i] = cc->ModReduce(strVector1[s][i]); // Level 14
		}
	}

	for (size_t i = 0; i < S.size(); i++)
		S.clear();
	S.clear();

	for (size_t i = 0; i < cX1.size(); i++)
		cX1.clear();
	cX1.clear();

	computation5Time = TOC(t);

	TIC(t);

	// Compute str * str
	std::vector<std::vector<Ciphertext<DCRTPoly>>> invVarD(sizeS);

	for (size_t s = 0; s < sizeS; s++)
		invVarD[s] = std::vector<Ciphertext<DCRTPoly>>(N);

	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for (size_t i = 0; i < N; i++)
		{
			invVarD[s][i] = cc->EvalMultNoRelin(strVector1[s][i],strVector1[s][i]); // Level 14
			//invVarD[i] = cc->ModReduce(invVarD[i]); // Level 15
		}
	}

	// Compute d*d
	auto cd12 = cc->LevelReduce(cd,nullptr,1); // Level 14
	auto cd2 = cc->EvalMult(cd12,cd12);
	cd2 = cc->ModReduce(cd2); // Level 15

	// Compute (w^T) (str*str)
#pragma omp parallel for
	for (size_t i = 0; i < N; i++)
	{
		cWVector[i] = cc->LevelReduce(cWVector[i],nullptr,3); // Level 14
                //scaling down
		auto temp = cc->EvalMult(cWVector[i],double(scalingFactorD));
		temp = cc->ModReduce(temp); // Level 15
		//auto temp = cc->EvalMultNoRelin(cd2,cWVector[i]);
		for (size_t s = 0; s < sizeS; s++) {
			invVarD[s][i] = cc->EvalMultNoRelin(temp,invVarD[s][i]);
		}
	}

	std::vector<Ciphertext<DCRTPoly>> invVar(sizeS);

	for (size_t s = 0; s < sizeS; s++)
		invVar[s] = BinaryTreeAdd(invVarD[s]);

	for (size_t s = 0; s < sizeS; s++)
		invVarD[s].clear();
	invVarD.clear();

	computation6Time = TOC(t);

	TIC(t);

	CompressEvalKeys(*rotKeysConv,4);

	// Compute w*ztr
	vector<Ciphertext<DCRTPoly>> beta1Arr(numCt);
	for (size_t r=0; r<numCt; r++){
		cWArr[r] = cc->LevelReduce(cWArr[r],nullptr,3); //Level 12

		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		size_t NPow2cur = 1<<(size_t)std::ceil(log2(N1));

		// clears the mask to prepare for the conversion
		std::vector<std::complex<double>> maskW2(m/4);
		for (size_t v = 0; v < NPow2cur*k*k; v++)
			maskW2[v] = scalingFactorN;
		for (size_t v = NPow2cur*k*k; v < m/4; v++)
			maskW2[v] = 0;
		Plaintext plaintextW2 = cc->MakeCKKSPackedPlaintext(maskW2,1,0,nullptr,m/4);
		auto cWConv2 = cc->EvalMult(cWArr[r],plaintextW2);

		cWConv2 = cc->ModReduce(cWConv2);//Level 13

		auto beta1 = cc->EvalMult(cWConv2,cztrArr[r]);

		for (size_t j = NPow2cur*k*k; j < m/4; j=j*2 ) {
			beta1 = cc->EvalAdd(beta1,cc->GetScheme()->EvalAtIndex(beta1,m/4-j,*rotKeysConv));
		}

		beta1Arr[r] = cc->ModReduce(beta1); //Level 14

	}

	CompressEvalKeys(*rotKeysConv,1);
	//CompressEvalKeys(*rotKeysBabyGiant,6);

	vector<Ciphertext<DCRTPoly>> betaVector;
	for (size_t r=0; r<numCt; r++){
		size_t N1;
		if ((r+1)*n < N*k2)
			N1 = n/(k2);
		else
			N1 = N - r*n/k2;
		auto temp = SplitIntoSingle(beta1Arr[r], N1, k, *rotKeysConv); //Level 15
		betaVector.insert(betaVector.end(),temp->begin(),temp->end());
	}

	rotKeysConv->clear();

	computation7Time = TOC(t);

	TIC(t);

	// Compute (w*ztr)^T str
	for (size_t s = 0; s < sizeS; s++) {
#pragma omp parallel for
		for (size_t i = 0; i < N; i++)
		{
			strVector1[s][i] = cc->LevelReduce(strVector1[s][i],nullptr,1);// Level 15
			strVector1[s][i] = cc->EvalMultNoRelin(betaVector[i],strVector1[s][i]);
		}
	}

	betaVector.clear();

	std::vector<Ciphertext<DCRTPoly>> beta(sizeS);

	for (size_t s = 0; s < sizeS; s++)
		beta[s] = BinaryTreeAdd(strVector1[s]);

	for (size_t s = 0; s < sizeS; s++)
		strVector1[s].clear();
	strVector1.clear();

	computation8Time = TOC(t);

	std::vector<Plaintext> pInvVar(sizeS);
	std::vector<Plaintext> pBeta(sizeS);
	Plaintext pD;

	TIC(t);
	cc->Decrypt(keyPair.secretKey, cd2 , &pD);
	for (size_t s = 0; s < sizeS; s++) {
		cc->Decrypt(keyPair.secretKey, invVar[s] , &(pInvVar[s]));
		cc->Decrypt(keyPair.secretKey, beta[s] , &(pBeta[s]));
	}

	decryptionTime = TOC(t);

	std::vector<double> zval(headersS.size());
	std::vector<double> pval(headersS.size());
	std::vector<double> betaval(headersS.size());
	std::vector<std::complex<double>> num(headersS.size());
	std::vector<std::complex<double>> den(headersS.size());

	for (size_t s = 0; s < sizeS; s++) {
		for (size_t i = 0; i < m/4; i++) {
			if (s*m/4 + i < headersS.size()) {
				num[s*m/4 + i] = pow(scalingFactor,-1)*pow(scalingFactorN,-1)*pBeta[s]->GetCKKSPackedValue()[i];
				den[s*m/4 + i] = pow(scalingFactor,-2)*pow(scalingFactorD,-1)*pInvVar[s]->GetCKKSPackedValue()[i];
				betaval[s*m/4 + i] = num[s*m/4 + i].real()/den[s*m/4 + i].real();
				zval[s*m/4 + i] = num[s*m/4 + i].real()/sqrt(den[s*m/4 + i].real()*pD->GetCKKSPackedValue()[0].real());
				pval[s*m/4 + i] = 2*sf(abs(zval[s*m/4 + i]));
				if (pval[s*m/4 + i] == 0)
					pval[s*m/4 + i] = BS(zval[s*m/4 + i]);
			}
		}
	}

    ofstream myfile;
    myfile.open(SNPDir + "/" + pValue);
    myfile.precision(10);
    for(uint32_t i = 0; i < headersS.size(); i++) {
    	myfile << headersS[i] << "\t" << pval[i] << std::endl;
    }
    myfile.close();

    ofstream myfilez;
    myfilez.open(SNPDir + "/" + "zvalue.txt");
    myfilez.precision(10);
    for(uint32_t i = 0; i < headersS.size(); i++) {
    	myfilez << headersS[i] << "\t" << zval[i] << std::endl;
    }
    myfilez.close();

    ofstream myfileb;
    myfileb.open(SNPDir + "/" + "betavalue.txt");
    myfileb.precision(10);
    for(uint32_t i = 0; i < headersS.size(); i++) {
    	myfileb << headersS[i] << "\t" << betaval[i] << std::endl;
    }
    myfileb.close();

    ofstream myfilenum;
    myfilenum.open(SNPDir + "/" + "num.txt");
    myfilenum.precision(10);
    for(uint32_t i = 0; i < headersS.size(); i++) {
    	myfilenum << headersS[i] << "\t" << num[i] << std::endl;
    }
    myfilenum.close();

    ofstream myfileden;
    myfileden.open(SNPDir + "/" + "den.txt");
    myfileden.precision(10);
    for(uint32_t i = 0; i < headersS.size(); i++) {
    	myfileden << headersS[i] << "\t" << den[i] << std::endl;
    }
    myfileden.close();

	computationTime = computation1Time + computation2Time + computation3Time + computation4Time +
			computation5Time + computation6Time + computation7Time + computation8Time;

	std::cout << "\nKey Generation Time: \t\t" << keyGenTime/1000 << " s" << std::endl;
	std::cout << "Encoding and Encryption Time: \t" << encryptionTime/1000 << " s" << std::endl;
	std::cout << "Computation Time: \t\t" << computationTime/1000 << " s" << std::endl;
	std::cout << "Decryption & Decoding Time: \t" << decryptionTime/1000 << " s" << std::endl;

	endToEndTime = TOC(tAll);

    std::cout << "\nEnd-to-end Runtime: \t\t" << endToEndTime/1000 << " s" << std::endl;

    ofstream myfileRuntime;
    myfileRuntime.open(SNPDir + "/" + Runtime);
    myfileRuntime << "Key Generation Time: \t\t" << keyGenTime/1000 << " s" << std::endl;
    myfileRuntime << "Encoding and Encryption Time: \t" << encryptionTime/1000 << " s" << std::endl;
    myfileRuntime << "Computation Time: \t\t" << computationTime/1000 << " s" << std::endl;
    myfileRuntime << "Decryption & Decoding Time: \t" << decryptionTime/1000 << " s" << std::endl;
    myfileRuntime << "End-to-end Runtime: \t\t" << endToEndTime/1000 << " s" << std::endl;
    myfileRuntime.close();

}

Ciphertext<DCRTPoly> zExpand(const Ciphertext<DCRTPoly> p, const Ciphertext<DCRTPoly> y) {

	CryptoContext<DCRTPoly> cc = p->GetCryptoContext();

	//Compute p-0.5
	auto pAdj = cc->EvalSub(p,double(0.25));
	pAdj = cc->EvalSub(pAdj,double(0.25));

	//Compute (p-0.5)^2; level 4
	auto p2 = cc->EvalMult(pAdj,pAdj);
	p2 = cc->ModReduce(p2);

	//Compute (p-0.5)^3; level 5
	auto p1 = cc->LevelReduce(pAdj,nullptr);
	auto p3 = cc->EvalMult(p2,p1);
	p3 = cc->ModReduce(p3);

	//Compute (p-0.5)^4; level 5
	auto p4 = cc->EvalMult(p2,p2);
	p4 = cc->ModReduce(p4);

	//Compute (p-0.5)^5; level 6
	p1 = cc->LevelReduce(p1,nullptr);
	auto p5 = cc->EvalMult(p4,p1);
	p5 = cc->ModReduce(p5);

	//Compute (p-0.5)^6; level 6
	p2 = cc->LevelReduce(p2,nullptr);
	auto p6 = cc->EvalMult(p4,p2);
	p6 = cc->ModReduce(p6);

	//Compute (p-0.5)^7; level 6
	auto p7 = cc->EvalMult(p4,p3);
	p7 = cc->ModReduce(p7);

	//Compute (p-0.5)^8; level 6
	auto p8 = cc->EvalMult(p4,p4);
	p8 = cc->ModReduce(p8);

	//Compute -2 + 4y
	auto factor = cc->EvalMult(y,4);
	factor = cc->ModReduce(factor); //level 1
	factor = cc->EvalSub(factor,2);
	auto t0 = cc->LevelReduce(factor,nullptr,6); //level 7

	//Compute (-8 + 16y)*(p-0.5)^2
	auto t1 = cc->EvalMult(factor,4);
	t1 = cc->ModReduce(t1); // level 2
	t1 = cc->LevelReduce(t1,nullptr,3); //level 5
	t1 = cc->EvalMult(t1,p2);
	t1 = cc->ModReduce(t1); //level 6
	t1 = cc->LevelReduce(t1,nullptr); //level 7

	//Compute (32/3)*(p-0.5)^3
	auto t2 = cc->EvalMult(p3,double(32/3));
	t2 = cc->ModReduce(t2); // level 6
	t2 = cc->LevelReduce(t2,nullptr); //level 7

	//Compute (-32 + 64y)*(p-0.5)^4
	auto t3 = cc->EvalMult(factor,16);
	t3 = cc->ModReduce(t3); // level 2
	t3 = cc->LevelReduce(t3,nullptr,3); //level 5
	t3 = cc->EvalMult(t3,p4);
	t3 = cc->ModReduce(t3); //level 6
	t3 = cc->LevelReduce(t3,nullptr); //level 7

	//Compute (256/5)*(p-0.5)^5
	auto t4 = cc->EvalMult(p5,double(256/5));
	t4 = cc->ModReduce(t4); // level 7

	//Compute (-128 + 256y)*(p-0.5)^6
	auto t5 = cc->EvalMult(factor,64);
	t5 = cc->ModReduce(t5); // level 2
	t5 = cc->LevelReduce(t5,nullptr,4); //level 6
	t5 = cc->EvalMult(t5,p6);
	t5 = cc->ModReduce(t5); //level 7

	//Compute (1536/7)*(p-0.5)^7
	auto t6 = cc->EvalMult(p7,double(1536/7));
	t6 = cc->ModReduce(t6); // level 7

	//Compute (-512 + 1024y)*(p-0.5)^8
	auto t7 = cc->EvalMult(factor,256);
	t7 = cc->ModReduce(t7); // level 2
	t7 = cc->LevelReduce(t7,nullptr,4); //level 6
	t7 = cc->EvalMult(t7,p8);
	t7 = cc->ModReduce(t7); //level 7

	auto z = cc->EvalAdd(t0,t1);
	z = cc->EvalSub(z,t2);
	z = cc->EvalAdd(z,t3);
	z = cc->EvalSub(z,t4);
	z = cc->EvalAdd(z,t5);
	z = cc->EvalSub(z,t6);
	z = cc->EvalAdd(z,t7);

	return z;

}

shared_ptr<std::vector<std::vector<Ciphertext<DCRTPoly>>>>  MatrixInverse(const Ciphertext<DCRTPoly> cM, size_t k,
		CiphertextImpl<DCRTPoly> &B, CiphertextImpl<DCRTPoly> &d, const std::map<usint, EvalKey<DCRTPoly>> &evalSum,
		const std::map<usint, EvalKey<DCRTPoly>> &rotKeys, const std::map<usint, EvalKey<DCRTPoly>> &evalSumRows) {

	auto cc = cM->GetCryptoContext();

	const shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParams = cM->GetCryptoParameters();
	const auto elementParams = cryptoParams->GetElementParams();
	usint m = elementParams->GetCyclotomicOrder();

	size_t kSquare = k*k;

	std::vector<std::complex<double>> mask(m/4);
	for (size_t i = 0; i < mask.size(); i++)
	{
		if (i % kSquare == 0)
			mask[i] = 1;
		else
			mask[i] = 0;
	}

	Plaintext plaintext = cc->MakeCKKSPackedPlaintext(mask,1,0,nullptr,m/4);

	std::vector<Ciphertext<DCRTPoly>> cMRotations(k*k-1);

	auto precomputedcM = cc->EvalFastRotationPrecompute(cM);

#pragma omp parallel for
	for (size_t i = 1; i < k*k; i++) {

		usint autoIndex = FindAutomorphismIndex2nComplex(i,m);

		if (i < 3)
			cMRotations[i-1] = HoistedAutomorphism(evalSum.find(autoIndex)->second,cM,precomputedcM,autoIndex);
		else if ((i == 4) || (i==8))
			cMRotations[i-1] = HoistedAutomorphism(evalSumRows.find(autoIndex)->second,cM,precomputedcM,autoIndex);
		else
			cMRotations[i-1] = HoistedAutomorphism(rotKeys.find(autoIndex)->second,cM,precomputedcM,autoIndex);

		cMRotations[i-1] = cc->ModReduce(cMRotations[i-1]);
		// clear all values that are not used
		cMRotations[i-1] = cc->EvalMult(cMRotations[i-1],plaintext);
		cMRotations[i-1] = cc->ModReduce(cMRotations[i-1]);
	}

	auto cMReduced = cc->ModReduce(cM);
	cMReduced = cc->LevelReduce(cMReduced,nullptr);

	auto a11a22 = cc->EvalMult(cMReduced,cMRotations[4]);
	a11a22 = cc->ModReduce(a11a22);

	auto a11a23 = cc->EvalMult(cMReduced,cMRotations[5]);
	a11a23 = cc->ModReduce(a11a23);

	auto a11a24 = cc->EvalMult(cMReduced,cMRotations[6]);
	a11a24 = cc->ModReduce(a11a24);

	auto a12a12 = cc->EvalMult(cMRotations[0],cMRotations[0]);
	a12a12 = cc->ModReduce(a12a12);

	auto a12a13 = cc->EvalMult(cMRotations[0],cMRotations[1]);
	a12a13 = cc->ModReduce(a12a13);

	auto a12a14 = cc->EvalMult(cMRotations[0],cMRotations[2]);
	a12a14 = cc->ModReduce(a12a14);

	auto a13a13 = cc->EvalMult(cMRotations[1],cMRotations[1]);
	a13a13 = cc->ModReduce(a13a13);

	auto a13a14 = cc->EvalMult(cMRotations[1],cMRotations[2]);
	a13a14 = cc->ModReduce(a13a14);

	auto a14a14 = cc->EvalMult(cMRotations[2],cMRotations[2]);
	a14a14 = cc->ModReduce(a14a14);

	auto a22a33 = cc->EvalMult(cMRotations[4],cMRotations[9]);
	a22a33 = cc->ModReduce(a22a33);

	auto a22a34 = cc->EvalMult(cMRotations[4],cMRotations[10]);
	a22a34 = cc->ModReduce(a22a34);

	auto a22a44 = cc->EvalMult(cMRotations[4],cMRotations[14]);
	a22a44 = cc->ModReduce(a22a44);

	auto a23a23 = cc->EvalMult(cMRotations[5],cMRotations[5]);
	a23a23 = cc->ModReduce(a23a23);

	auto a23a24 = cc->EvalMult(cMRotations[5],cMRotations[6]);
	a23a24 = cc->ModReduce(a23a24);

	auto a23a34 = cc->EvalMult(cMRotations[5],cMRotations[10]);
	a23a34 = cc->ModReduce(a23a34);

	auto a23a44 = cc->EvalMult(cMRotations[5],cMRotations[14]);
	a23a44 = cc->ModReduce(a23a44);

	auto a24a24 = cc->EvalMult(cMRotations[6],cMRotations[6]);
	a24a24 = cc->ModReduce(a24a24);

	auto a24a33 = cc->EvalMult(cMRotations[6],cMRotations[9]);
	a24a33 = cc->ModReduce(a24a33);

	auto a24a34 = cc->EvalMult(cMRotations[6],cMRotations[10]);
	a24a34 = cc->ModReduce(a24a34);

	auto a33a44 = cc->EvalMult(cMRotations[9],cMRotations[14]);
	a33a44 = cc->ModReduce(a33a44);

	auto a34a34 = cc->EvalMult(cMRotations[10],cMRotations[10]);
	a34a34 = cc->ModReduce(a34a34);

/*
 * det =  a[1,4]*a[1,4]*a[2,3]*a[2,3] - 2*a[1,3]*a[1,4]*a[2,3]*a[2,4] +   a[1,3]*a[1,3]*a[2,4]*a[2,4] -
         a[1,4]*a[1,4]*a[2,2]*a[3,3] + 2*a[1,2]*a[1,4]*a[2,4]*a[3,3] -   a[1,1]*a[2,4]*a[2,4]*a[3,3] +
       2*a[1,3]*a[1,4]*a[2,2]*a[3,4] - 2*a[1,2]*a[1,4]*a[2,3]*a[3,4] - 2*a[1,2]*a[1,3]*a[2,4]*a[3,4] +
       2*a[1,1]*a[2,3]*a[2,4]*a[3,4] +   a[1,2]*a[1,2]*a[3,4]*a[3,4] -   a[1,1]*a[2,2]*a[3,4]*a[3,4] -
         a[1,3]*a[1,3]*a[2,2]*a[4,4] + 2*a[1,2]*a[1,3]*a[2,3]*a[4,4] -   a[1,1]*a[2,3]*a[2,3]*a[4,4] -
         a[1,2]*a[1,2]*a[3,3]*a[4,4] +   a[1,1]*a[2,2]*a[3,3]*a[4,4]
 */

	auto cd = cc->EvalMultNoRelin(a14a14,a23a23);

	auto temp = cc->EvalMultNoRelin(a13a14,a23a24);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalSub(cd,temp);
	cd = cc->EvalAdd(cd,cc->EvalMultNoRelin(a13a13,a24a24));
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a14a14,a22a33));

	//std::cerr << *cd << std::endl;

	temp = cc->EvalMultNoRelin(a12a14,a24a33);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalAdd(cd,temp);
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a11a24,a24a33));

	temp = cc->EvalMultNoRelin(a13a14,a22a34);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalAdd(cd,temp);

	temp = cc->EvalMultNoRelin(a12a14,a23a34);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalSub(cd,temp);

	temp = cc->EvalMultNoRelin(a12a13,a24a34);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalSub(cd,temp);

	temp = cc->EvalMultNoRelin(a11a23,a24a34);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalAdd(cd,temp);
	cd = cc->EvalAdd(cd,cc->EvalMultNoRelin(a12a12,a34a34));
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a11a22,a34a34));
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a13a13,a22a44));

	temp = cc->EvalMultNoRelin(a12a13,a23a44);
	temp = cc->EvalAdd(temp,temp);

	cd = cc->EvalAdd(cd,temp);
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a11a23,a23a44));
	cd = cc->EvalSub(cd,cc->EvalMultNoRelin(a12a12,a33a44));
	cd = cc->EvalAdd(cd,cc->EvalMultNoRelin(a11a22,a33a44));

	cd = cc->Relinearize(cd);

	Ciphertext<DCRTPoly> cdOld;

	for (size_t i = 1; i < k*k; i = i*2)
	{
		cdOld = cd;
		if (i < 3)
			cd = cc->GetScheme()->EvalAtIndex(cdOld,i,evalSum);
		else if ( (i == 4) || (i == 8) )
			cd = cc->GetScheme()->EvalAtIndex(cdOld,i,evalSumRows);
		else
			cd = cc->GetScheme()->EvalAtIndex(cdOld,i,rotKeys);
		cd = cc->EvalAdd(cdOld,cd);
	}

	cd = cc->ModReduce(cd);

	d = *cd;

/*

# Adjoint of a 4 by 4 symmetric matrix
adjoin_4by4_sim_matrix <- function(a){
  b11 = -a[4,4]*a[2,3]*a[2,3] + 2*a[2,4]*a[3,4]*a[2,3] - a[2,2]*a[3,4]*a[3,4] - a[2,4]*a[2,4]*a[3,3] + a[2,2]*a[3,3]*a[4,4]
  b12 =  a[1,2]*a[3,4]*a[3,4] -   a[1,4]*a[2,3]*a[3,4] - a[1,3]*a[2,4]*a[3,4] + a[1,4]*a[2,4]*a[3,3] + a[1,3]*a[2,3]*a[4,4] - a[1,2]*a[3,3]*a[4,4]
  b13 =  a[1,3]*a[2,4]*a[2,4] -   a[1,4]*a[2,3]*a[2,4] - a[1,2]*a[3,4]*a[2,4] + a[1,4]*a[2,2]*a[3,4] - a[1,3]*a[2,2]*a[4,4] + a[1,2]*a[2,3]*a[4,4]
  b14 =  a[1,4]*a[2,3]*a[2,3] -   a[1,3]*a[2,4]*a[2,3] - a[1,2]*a[3,4]*a[2,3] - a[1,4]*a[2,2]*a[3,3] + a[1,2]*a[2,4]*a[3,3] + a[1,3]*a[2,2]*a[3,4]

  b22 = -a[4,4]*a[1,3]*a[1,3] + 2*a[1,4]*a[3,4]*a[1,3] - a[1,1]*a[3,4]*a[3,4] - a[1,4]*a[1,4]*a[3,3] + a[1,1]*a[3,3]*a[4,4]
  b23 =  a[2,3]*a[1,4]*a[1,4] -   a[1,3]*a[2,4]*a[1,4] - a[1,2]*a[3,4]*a[1,4] + a[1,1]*a[2,4]*a[3,4] + a[1,2]*a[1,3]*a[4,4] - a[1,1]*a[2,3]*a[4,4]
  b24 =  a[2,4]*a[1,3]*a[1,3] -   a[1,4]*a[2,3]*a[1,3] - a[1,2]*a[3,4]*a[1,3] + a[1,2]*a[1,4]*a[3,3] - a[1,1]*a[2,4]*a[3,3] + a[1,1]*a[2,3]*a[3,4]

  b33 = -a[4,4]*a[1,2]*a[1,2] + 2*a[1,4]*a[2,4]*a[1,2] - a[1,1]*a[2,4]*a[2,4] - a[1,4]*a[1,4]*a[2,2] + a[1,1]*a[2,2]*a[4,4]
  b34 =  a[3,4]*a[1,2]*a[1,2] -   a[1,4]*a[2,3]*a[1,2] - a[1,3]*a[2,4]*a[1,2] + a[1,3]*a[1,4]*a[2,2] + a[1,1]*a[2,3]*a[2,4] - a[1,1]*a[2,2]*a[3,4]

  b44 = -a[3,3]*a[1,2]*a[1,2] + 2*a[1,3]*a[2,3]*a[1,2] - a[1,1]*a[2,3]*a[2,3] - a[1,3]*a[1,3]*a[2,2] + a[1,1]*a[2,2]*a[3,3]

  b <- matrix(c(b11,b12,b13,b14,b12,b22,b23,b24,b13,b23,b33,b34,b14,b24,b34,b44), ncol=4, byrow=TRUE)

*/

	for (size_t i = 1; i < k*k; i++) {
		cMRotations[i-1] = cc->LevelReduce(cMRotations[i-1],nullptr);
	}
	cMReduced = cc->LevelReduce(cMReduced,nullptr);

	temp = cc->EvalMultNoRelin(cMRotations[6],a23a34);
	temp = cc->EvalAdd(temp,temp);

	auto b11 = temp;

	// We can a binary tree approach here and below
	b11 = cc->EvalSub(b11,cc->EvalMultNoRelin(cMRotations[14],a23a23));
	b11 = cc->EvalSub(b11,cc->EvalMultNoRelin(cMRotations[4],a34a34));
	b11 = cc->EvalSub(b11,cc->EvalMultNoRelin(cMRotations[9],a24a24));
	b11 = cc->EvalAdd(b11,cc->EvalMultNoRelin(cMRotations[4],a33a44));

	b11 = cc->Relinearize(b11);

	auto b12 = cc->EvalMultNoRelin(cMRotations[0],a34a34);
	b12 = cc->EvalSub(b12,cc->EvalMultNoRelin(cMRotations[2],a23a34));
	b12 = cc->EvalSub(b12,cc->EvalMultNoRelin(cMRotations[1],a24a34));
	b12 = cc->EvalAdd(b12,cc->EvalMultNoRelin(cMRotations[2],a24a33));
	b12 = cc->EvalAdd(b12,cc->EvalMultNoRelin(cMRotations[1],a23a44));
	b12 = cc->EvalSub(b12,cc->EvalMultNoRelin(cMRotations[0],a33a44));

	b12 = cc->Relinearize(b12);

	auto b13 = cc->EvalMultNoRelin(cMRotations[1],a24a24);
	b13 = cc->EvalSub(b13,cc->EvalMultNoRelin(cMRotations[2],a23a24));
	b13 = cc->EvalSub(b13,cc->EvalMultNoRelin(cMRotations[0],a24a34));
	b13 = cc->EvalAdd(b13,cc->EvalMultNoRelin(cMRotations[2],a22a34));
	b13 = cc->EvalSub(b13,cc->EvalMultNoRelin(cMRotations[1],a22a44));
	b13 = cc->EvalAdd(b13,cc->EvalMultNoRelin(cMRotations[0],a23a44));

	b13 = cc->Relinearize(b13);

	auto b14 = cc->EvalMultNoRelin(cMRotations[2],a23a23);
	b14 = cc->EvalSub(b14,cc->EvalMultNoRelin(cMRotations[1],a23a24));
	b14 = cc->EvalSub(b14,cc->EvalMultNoRelin(cMRotations[0],a23a34));
	b14 = cc->EvalSub(b14,cc->EvalMultNoRelin(cMRotations[2],a22a33));
	b14 = cc->EvalAdd(b14,cc->EvalMultNoRelin(cMRotations[0],a24a33));
	b14 = cc->EvalAdd(b14,cc->EvalMultNoRelin(cMRotations[1],a22a34));

	b14 = cc->Relinearize(b14);

	temp = cc->EvalMultNoRelin(cMRotations[10],a13a14);
	temp = cc->EvalAdd(temp,temp);

	auto b22 = temp;

	b22 = cc->EvalSub(b22,cc->EvalMultNoRelin(cMRotations[14],a13a13));
	b22 = cc->EvalSub(b22,cc->EvalMultNoRelin(cMReduced,a34a34));
	b22 = cc->EvalSub(b22,cc->EvalMultNoRelin(cMRotations[9],a14a14));
	b22 = cc->EvalAdd(b22,cc->EvalMultNoRelin(cMReduced,a33a44));

	b22 = cc->Relinearize(b22);

	auto b23 = cc->EvalMultNoRelin(cMRotations[5],a14a14);
	b23 = cc->EvalSub(b23,cc->EvalMultNoRelin(cMRotations[6],a13a14));
	b23 = cc->EvalSub(b23,cc->EvalMultNoRelin(cMRotations[10],a12a14));
	b23 = cc->EvalAdd(b23,cc->EvalMultNoRelin(cMReduced,a24a34));
	b23 = cc->EvalAdd(b23,cc->EvalMultNoRelin(cMRotations[14],a12a13));
	b23 = cc->EvalSub(b23,cc->EvalMultNoRelin(cMRotations[14],a11a23));

	b23 = cc->Relinearize(b23);

	// We can a binary tree approach here
	auto b24 = cc->EvalMultNoRelin(cMRotations[6],a13a13);
	b24 = cc->EvalSub(b24,cc->EvalMultNoRelin(cMRotations[5],a13a14));
	b24 = cc->EvalSub(b24,cc->EvalMultNoRelin(cMRotations[10],a12a13));
	b24 = cc->EvalAdd(b24,cc->EvalMultNoRelin(cMRotations[9],a12a14));
	b24 = cc->EvalSub(b24,cc->EvalMultNoRelin(cMRotations[9],a11a24));
	b24 = cc->EvalAdd(b24,cc->EvalMultNoRelin(cMRotations[10],a11a23));

	b24 = cc->Relinearize(b24);

	temp = cc->EvalMultNoRelin(cMRotations[6],a12a14);
	temp = cc->EvalAdd(temp,temp);

	auto b33 = temp;

	b33 = cc->EvalSub(b33,cc->EvalMultNoRelin(cMRotations[14],a12a12));
	b33 = cc->EvalSub(b33,cc->EvalMultNoRelin(cMReduced,a24a24));
	b33 = cc->EvalSub(b33,cc->EvalMultNoRelin(cMRotations[4],a14a14));
	b33 = cc->EvalAdd(b33,cc->EvalMultNoRelin(cMReduced,a22a44));

	b33 = cc->Relinearize(b33);

	// We can a binary tree approach here
	auto b34 = cc->EvalMultNoRelin(cMRotations[10],a12a12);
	b34 = cc->EvalSub(b34,cc->EvalMultNoRelin(cMRotations[5],a12a14));
	b34 = cc->EvalSub(b34,cc->EvalMultNoRelin(cMRotations[6],a12a13));
	b34 = cc->EvalAdd(b34,cc->EvalMultNoRelin(cMRotations[4],a13a14));
	b34 = cc->EvalAdd(b34,cc->EvalMultNoRelin(cMReduced,a23a24));
	b34 = cc->EvalSub(b34,cc->EvalMultNoRelin(cMReduced,a22a34));

	b34 = cc->Relinearize(b34);

	temp = cc->EvalMultNoRelin(cMRotations[5],a12a13);
	temp = cc->EvalAdd(temp,temp);

	auto b44 = temp;

	// We can a binary tree approach here
	b44 = cc->EvalSub(b44,cc->EvalMultNoRelin(cMRotations[9],a12a12));
	b44 = cc->EvalSub(b44,cc->EvalMultNoRelin(cMReduced,a23a23));
	b44 = cc->EvalSub(b44,cc->EvalMultNoRelin(cMRotations[4],a13a13));
	b44 = cc->EvalAdd(b44,cc->EvalMultNoRelin(cMReduced,a22a33));

	b44 = cc->Relinearize(b44);

	shared_ptr<std::vector<std::vector<Ciphertext<DCRTPoly>>>> B1(new std::vector<std::vector<Ciphertext<DCRTPoly>>>(k));

	// We can use a binary tree approach here
	auto b = cc->EvalAdd(b11,cc->GetScheme()->EvalAtIndex(b12,k*k-1,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b13,k*k-2,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b14,k*k-3,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b12,k*k-4,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b22,k*k-5,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b23,k*k-6,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b24,k*k-7,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b13,k*k-8,evalSumRows));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b23,k*k-9,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b33,k*k-10,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b34,k*k-11,rotKeys));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b14,k*k-12,evalSumRows));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b24,k*k-13,rotKeys));
	//b = cc->EvalAdd(b,cc->EvalAtIndex(b34,k*k-14));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b34,k*k-14,evalSum));
	//b = cc->EvalAdd(b,cc->EvalAtIndex(b44,k*k-15));
	b = cc->EvalAdd(b,cc->GetScheme()->EvalAtIndex(b44,k*k-15,evalSum));

	b = cc->ModReduce(b);

	size_t k2 = k*k;

	for (size_t i = 0; i<k; i++)
		(*B1)[i] = std::vector<Ciphertext<DCRTPoly>>(k);

	(*B1)[0][0] = cc->ModReduce(CloneCiphertext(b11,k2,rotKeys, evalSumRows));
	(*B1)[0][1] = cc->ModReduce(CloneCiphertext(b12,k2,rotKeys, evalSumRows));
	(*B1)[0][2] = cc->ModReduce(CloneCiphertext(b13,k2,rotKeys, evalSumRows));
	(*B1)[0][3] = cc->ModReduce(CloneCiphertext(b14,k2,rotKeys, evalSumRows));
	(*B1)[1][0] = cc->ModReduce(CloneCiphertext(b12,k2,rotKeys, evalSumRows));
	(*B1)[1][1] = cc->ModReduce(CloneCiphertext(b22,k2,rotKeys, evalSumRows));
	(*B1)[1][2] = cc->ModReduce(CloneCiphertext(b23,k2,rotKeys, evalSumRows));
	(*B1)[1][3] = cc->ModReduce(CloneCiphertext(b24,k2,rotKeys, evalSumRows));
	(*B1)[2][0] = cc->ModReduce(CloneCiphertext(b13,k2,rotKeys, evalSumRows));
	(*B1)[2][1] = cc->ModReduce(CloneCiphertext(b23,k2,rotKeys, evalSumRows));
	(*B1)[2][2] = cc->ModReduce(CloneCiphertext(b33,k2,rotKeys, evalSumRows));
	(*B1)[2][3] = cc->ModReduce(CloneCiphertext(b34,k2,rotKeys, evalSumRows));
	(*B1)[3][0] = cc->ModReduce(CloneCiphertext(b14,k2,rotKeys, evalSumRows));
	(*B1)[3][1] = cc->ModReduce(CloneCiphertext(b24,k2,rotKeys, evalSumRows));
	(*B1)[3][2] = cc->ModReduce(CloneCiphertext(b34,k2,rotKeys, evalSumRows));
	(*B1)[3][3] = cc->ModReduce(CloneCiphertext(b44,k2,rotKeys, evalSumRows));

	B = *b;

	return B1;

}

Ciphertext<DCRTPoly> CloneCiphertext(const Ciphertext<DCRTPoly> ciphertext, size_t size,
		const std::map<usint,EvalKey<DCRTPoly>> &rotKeys, const std::map<usint,EvalKey<DCRTPoly>> &evalSumRows) {

	Ciphertext<DCRTPoly> answer(new CiphertextImpl<DCRTPoly>(*ciphertext));
	auto cc = ciphertext->GetCryptoContext();

	for (size_t i = 1; i < size; i = i*2) {
		if ( ((size-i) == 4) || ((size-i) == 8) )
			answer = cc->EvalAdd(answer,cc->GetScheme()->EvalAtIndex(answer,size-i,evalSumRows));
		else
			answer = cc->EvalAdd(answer,cc->GetScheme()->EvalAtIndex(answer,size-i,rotKeys));
	}

	return answer;

}

shared_ptr<std::vector<Ciphertext<DCRTPoly>>> SplitIntoSingle(const Ciphertext<DCRTPoly> c, size_t N, size_t k,
		const std::map<usint, EvalKey<DCRTPoly>> &rotKeys){

	auto cc = c->GetCryptoContext();

	const shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParams = c->GetCryptoParameters();
	const auto elementParams = cryptoParams->GetElementParams();
	usint m = elementParams->GetCyclotomicOrder();

	shared_ptr<std::vector<Ciphertext<DCRTPoly>>> cVector(new std::vector<Ciphertext<DCRTPoly>>(N));

	size_t k2 = k*k;
	size_t NPow2k = k2*(1<<(size_t)std::ceil(log2(N)));

#pragma omp parallel for
	for (size_t i = 0; i < N; i++) {
		std::vector<std::complex<double>> mask(m/4);
		for (size_t v = 0; v < mask.size(); v++)
		{
			if (((v % (NPow2k)) >= i*k2) && ((v % (NPow2k)) < (i+1)*k2))
				mask[v] = 1;
			else
				mask[v] = 0;
		}
		Plaintext plaintext = cc->MakeCKKSPackedPlaintext(mask,1,0,nullptr,m/4);
		auto cTemp = cc->EvalMult(c,plaintext);
		for (size_t j = k2; j < NPow2k; j=j*2 ) {
			cTemp = cc->EvalAdd(cTemp,cc->GetScheme()->EvalAtIndex(cTemp,m/4-j,rotKeys));
		}
		(*cVector)[i] = cc->ModReduce(cTemp);
	}

	return cVector;

}

Ciphertext<DCRTPoly> BinaryTreeAdd(std::vector<Ciphertext<DCRTPoly>> &vector) {

	auto cc = vector[0]->GetCryptoContext();

	for(size_t j = 1; j < vector.size(); j=j*2) {
		for(size_t i = 0; i<vector.size(); i = i + 2*j) {
			if ((i+j)<vector.size())
				vector[i] = cc->EvalAdd(vector[i],vector[i+j]);
		}
	}

	Ciphertext<DCRTPoly> result(new CiphertextImpl<DCRTPoly>(*(vector[0])));

	return result;

}

void ReadSNPFile(vector<string>& headers, std::vector<std::vector<double>> & dataColumns, std::vector<std::vector<double>> &x, std::vector<double> &y,
                 string dataFileName, size_t N, size_t M)
{

	uint32_t cols = 0;
	uint32_t xcols = 3;

	string fileName = dataFileName + ".csv";

	std::cerr << "file name = " << fileName << std::endl;

	ifstream file(fileName);
	string line, value;

	if(file.good()) {

		getline(file, line);
		cols = std::count(line.begin(), line.end(), ',') + 1;
		stringstream ss(line);
		vector<string> result;

		size_t tempcounter = 0;

		for(uint32_t i = 0; i < cols; i++) {
			string substr;
			getline(ss, substr, ',');
			if ((substr != "") && (i>4) && (i<M+5)) {
				headers.push_back(substr);
				tempcounter++;
			}
		}

		cols = tempcounter;

	}

	size_t counter = 0;
	while((file.good()) && (counter < N)) {
		getline(file, line);
		uint32_t curCols = std::count(line.begin(), line.end(), ',') + 1;
		//std::cout << numCols << std::endl;
		if (curCols > 2) {
			stringstream ss(line);
			std::vector<double> xrow(xcols);
			for(uint32_t i = 0; i < 5; i++) {
				string substr;
				getline(ss, substr, ',');
				if (i==1)
					y.push_back(std::stod(substr));
				else if (i>1)
					xrow[i-2] = std::stod(substr);
			}
			x.push_back(xrow);
			std::vector<double> row(cols);
			for(uint32_t i = 5; i < cols + 5; i++) {
				string substr;
				getline(ss, substr, ',');
				if (i < M+5)
				{
					double val;
					val = std::stod(substr);
					row[i-5] = val;
				}
			}
			dataColumns.push_back(row);
		}
		counter++;
	}

	file.close();

	 // insert the intercept
	for (size_t i = 0; i < x.size(); i++)
	{
		x[i].insert(x[i].begin(),double(1));
	}

    std::cout << "Read in data: ";
	std::cout << dataFileName << std::endl;
}

void CompressEvalKeys(std::map<usint, EvalKey<DCRTPoly>> &ek, size_t level) {

	std::map<usint, EvalKey<DCRTPoly>>::iterator it;

	for ( it = ek.begin(); it != ek.end(); it++ )
	{

		std::vector<DCRTPoly> b = it->second->GetBVector();
		std::vector<DCRTPoly> a = it->second->GetAVector();

		for (size_t k = 0; k < a.size(); k++) {
			a[k].DropLastElements(level);
			b[k].DropLastElements(level);
		}

		it->second->ClearKeys();

		it->second->SetAVector(std::move(a));
		it->second->SetBVector(std::move(b));

	}

}

Ciphertext<DCRTPoly> HoistedAutomorphism(const EvalKey<DCRTPoly> ek,
	ConstCiphertext<DCRTPoly> cipherText, const shared_ptr<vector<DCRTPoly>> digits, const usint index)
{


	Ciphertext<DCRTPoly> newCiphertext = cipherText->CloneEmpty();

	const shared_ptr<CryptoParametersCKKSRNS> cryptoParamsLWE = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ek->GetCryptoParameters());

	EvalKeyRelin<DCRTPoly> evalKey = std::static_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(ek);

	const std::vector<DCRTPoly> &c = cipherText->GetElements();

	std::vector<DCRTPoly> b = evalKey->GetBVector();
	std::vector<DCRTPoly> a = evalKey->GetAVector();

	size_t towersToDrop = b[0].GetParams()->GetParams().size() - c[0].GetParams()->GetParams().size();

	for (size_t k = 0; k < b.size(); k++) {
		a[k].DropLastElements(towersToDrop);
		b[k].DropLastElements(towersToDrop);
	}

	std::vector<DCRTPoly> digitsC2(*digits);

	DCRTPoly ct0(c[0]);
	DCRTPoly ct1;

	ct1 = digitsC2[0] * a[0];
	ct0 += digitsC2[0] * b[0];

	for (usint i = 1; i < digitsC2.size(); ++i)
	{
		ct0 += digitsC2[i] * b[i];
		ct1 += digitsC2[i] * a[i];
	}

	ct0 = ct0.AutomorphismTransform(index);
	ct1 = ct1.AutomorphismTransform(index);

	newCiphertext->SetElements({ ct0, ct1 });

	newCiphertext->SetNoiseScaleDeg(cipherText->GetNoiseScaleDeg());
	newCiphertext->SetLevel(cipherText->GetLevel());
	newCiphertext->SetScalingFactor(cipherText->GetScalingFactor());
	newCiphertext->SetSlots(cipherText->GetSlots());

	return newCiphertext;
}




