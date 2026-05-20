#ifndef _LOWMC_H_
#define _LOWMC_H_

#define N 64
#define P 1
#define R 62

const int D = N - 3 * P;
const int ATTACK_R0_BASE = 21;
const int ATTACK_R0_EXT = 5;
const int ATTACK_R0 = ATTACK_R0_BASE + ATTACK_R0_EXT;
const int ATTACK_R1 = 28;
const int ATTACK_T = 11;
const int ATTACK_R2 = 8;
const int ATTACK_ROUNDS = ATTACK_R0 + ATTACK_R1 + ATTACK_R2;

const int ddt[8][4] = {
	0,0,0,0,
	1,3,5,7,
	2,3,6,7,
	1,5,2,6,
	4,5,6,7,
	1,3,4,6,
	2,3,4,5,
	1,2,4,7
};

#include <string>
#include <vector>
using namespace std;

struct M {
	int r = N;//default value
	int c = N;//default value
	bool ma[256][300];
};

struct AttackRecord {
	vector<bool> p0, p1, c0, c1, key;
	vector<vector<bool> > SOut0, LOut0;
	vector<vector<bool> > SOutDiff, LOutDiff;
};

class LowMC {
private:
	vector<vector<bool> > CONS;
	M* L;
	M* IL;
	M* KF;
	M* IKF;
	//M WK;//whitening key
	M H1, Q1, GammaTr;
	M PP0, QQ1;
	vector<vector<unsigned long long>> DH;
	vector<bool> attackR0Diff;
public:
	LowMC();
	~LowMC();
	void initializePars(string filename,bool isKey,M* mat);

	//encryption
	void encryptionDetails(bool p0[], bool k[], bool c[], int rounds, vector<vector<bool> >& LOut, vector<vector<bool> >& SOut);

	//used for the algebraic attacks
	void findInputDifference(bool x[], bool inputDiff[]);
	
	void enumerateDiffForwards(bool x[]);
	void forwards(int current, int total, vector<vector<bool>> &Dh, vector<bool> &u, vector<vector<bool> >& nonlin, vector<vector<bool> >& lin);
	void enumerateDiffBackwards(bool x[]);
	void enumerateDiffBackwards(bool x[], unsigned long long cnt[]);
	void enumerateDiffBackwards(bool x[], unsigned long long cnt[], vector<AttackRecord>& records, const AttackRecord& baseRecord);
	void clearDifferenceTable();
	void backwards(unsigned long long cnt[],int current, int total, vector<vector<bool> >& nonlin, vector<vector<bool> >& lin);
	void backwards(unsigned long long cnt[], int current, int total, vector<vector<bool> >& nonlin, vector<vector<bool> >& lin, vector<AttackRecord>& records, const AttackRecord& baseRecord);

	void constructPQ(bool x[]);
	void storeSolutions(vector<vector<unsigned long long>>& Dh, M& P0P, vector <bool>& cbit);
	int onlineCheck(vector<vector<unsigned long long>>& Dh, M& Q1, M& P0, vector<bool>& ga, unsigned long long& checkedCandidates);
	int onlineCheck(vector<vector<unsigned long long>>& Dh, M& Q1, M& P0, vector<bool>& ga, unsigned long long& checkedCandidates, vector<AttackRecord>& records, const AttackRecord& baseRecord, vector<vector<bool> >& backNonlin, vector<vector<bool> >& backLin);
	//bool checkCorrectness(vector<bool>& gamma, vector<bool>& alpha, vector<bool>& u,int rounds,int total);

	//key recovery
	int keyRecovery(bool key[], bool c[], int r, vector<vector<bool> >& LOut, vector<vector<bool> >& SOut, vector<vector<bool> >& sOut, vector<vector<bool> > lOut);
	int keyRecoveryAttack(bool p0[], bool p1[], bool c0[], bool c1[], int r, vector<vector<bool> >& sOut, vector<vector<bool> > lOut, bool recoveredKey[]);
	void dynamicallyUpdateExpression( M& expLOut, M& expSOut, int in, int out);
	void extractEquations(M& eq, M& expSOut, int in, int out);
	int solveKey(M& eqSys, M& extraEq, int extraSize, bool testVec[], bool key[], bool p0[] = NULL, bool p1[] = NULL, bool c0[] = NULL, bool c1[] = NULL);
	void addLinear(M& qeq, int row, M& lin, int index);
	void addQuadratic(M& qeq, int row, M& fac0, int in0, M& fac1, int in1, int total);

	//search algebraic equations
	void searchAlgebraicEquations();
	void computeQuadraticEq(bool x[], bool row[]);
	void computeCubicEq(bool x[], bool row[]);
	
	//matrix operations
	void matrixMul(M& m, bool x[], bool y[]);
	void matrixMul(M& m, vector<bool> &x, vector<bool>& y);
	void matrixMul(M& m, bool x[]);
	void matrixMul(M& m1, M& m2, M& m3);
	void matrixMul(M& m1, M& m2);
	void clearMatrix(M& m);
	void matrixEq(M& src, M& dec);
	void gauss(M& eqSys, int col);
	void simplify(M& eqSys, int col);
	void storeSolutions(vector<vector<bool> >& sol, M& eqSys, int& solNum);

	//output functions
	void outputM(M mat);
	void outputArray(bool vec[],int size);
};

#endif
