#include "LowMC.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <ctime>
#include <random>
#include <cmath>
using namespace std;

const string COMMAND2_PATH_FILE = "command2_paths.txt";

vector<bool> arrayToVector(bool x[], int size) {
	vector<bool> v(size);
	for (int i = 0; i < size; i++) {
		v[i] = x[i];
	}
	return v;
}

vector<vector<bool> > diffOfStates(vector<vector<bool> >& a, vector<vector<bool> >& b) {
	vector<vector<bool> > diff(ATTACK_ROUNDS, vector<bool>(N, 0));
	for (int i = 0; i < ATTACK_ROUNDS; i++) {
		for (int j = 0; j < N; j++) {
			diff[i][j] = a[i][j] ^ b[i][j];
		}
	}
	return diff;
}

void writeBits(ofstream& out, const vector<bool>& bits) {
	for (int i = 0; i < bits.size(); i++) {
		out << bits[i];
	}
	out << endl;
}

bool readBits(ifstream& in, vector<bool>& bits, int size) {
	string s;
	if (!(in >> s)) {
		return false;
	}
	if (s.size() != size) {
		return false;
	}
	bits.assign(size, 0);
	for (int i = 0; i < size; i++) {
		bits[i] = (s[i] == '1');
	}
	return true;
}

void saveCommand2Records(const vector<AttackRecord>& records) {
	ofstream out(COMMAND2_PATH_FILE);
	out << "COUNT " << records.size() << endl;
	for (int i = 0; i < records.size(); i++) {
		out << "RECORD " << i << endl;
		out << "P0 "; writeBits(out, records[i].p0);
		out << "P1 "; writeBits(out, records[i].p1);
		out << "C0 "; writeBits(out, records[i].c0);
		out << "C1 "; writeBits(out, records[i].c1);
		out << "KEY "; writeBits(out, records[i].key);
		out << "SOUT0" << endl;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			writeBits(out, records[i].SOut0[r]);
		}
		out << "LOUT0" << endl;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			writeBits(out, records[i].LOut0[r]);
		}
		out << "SOUTDIFF" << endl;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			writeBits(out, records[i].SOutDiff[r]);
		}
		out << "LOUTDIFF" << endl;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			writeBits(out, records[i].LOutDiff[r]);
		}
		out << "END" << endl;
	}
	out.close();
}

bool loadCommand2Records(vector<AttackRecord>& records) {
	ifstream in(COMMAND2_PATH_FILE);
	if (!in) {
		return false;
	}
	string tag;
	int count = 0;
	if (!(in >> tag >> count) || tag != "COUNT") {
		return false;
	}
	records.clear();
	for (int i = 0; i < count; i++) {
		AttackRecord rec;
		int index = 0;
		if (!(in >> tag >> index) || tag != "RECORD") {
			return false;
		}
		if (!(in >> tag) || tag != "P0" || !readBits(in, rec.p0, N)) return false;
		if (!(in >> tag) || tag != "P1" || !readBits(in, rec.p1, N)) return false;
		if (!(in >> tag) || tag != "C0" || !readBits(in, rec.c0, N)) return false;
		if (!(in >> tag) || tag != "C1" || !readBits(in, rec.c1, N)) return false;
		if (!(in >> tag) || tag != "KEY" || !readBits(in, rec.key, N)) return false;

		rec.SOut0.assign(ATTACK_ROUNDS, vector<bool>(N, 0));
		rec.LOut0.assign(ATTACK_ROUNDS, vector<bool>(N, 0));
		rec.SOutDiff.assign(ATTACK_ROUNDS, vector<bool>(N, 0));
		rec.LOutDiff.assign(ATTACK_ROUNDS, vector<bool>(N, 0));

		if (!(in >> tag) || tag != "SOUT0") return false;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			if (!readBits(in, rec.SOut0[r], N)) return false;
		}
		if (!(in >> tag) || tag != "LOUT0") return false;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			if (!readBits(in, rec.LOut0[r], N)) return false;
		}
		if (!(in >> tag) || tag != "SOUTDIFF") return false;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			if (!readBits(in, rec.SOutDiff[r], N)) return false;
		}
		if (!(in >> tag) || tag != "LOUTDIFF") return false;
		for (int r = 0; r < ATTACK_ROUNDS; r++) {
			if (!readBits(in, rec.LOutDiff[r], N)) return false;
		}
		if (!(in >> tag) || tag != "END") {
			return false;
		}
		records.push_back(rec);
	}
	return true;
}

void vectorToArray(const vector<bool>& v, bool x[]) {
	for (int i = 0; i < v.size(); i++) {
		x[i] = v[i];
	}
}

void computeComplexity(int r1, int r2, int t, int e, int eqSize, int n, int r,int m) {
	double t0 = 1.86 * m * (r1 + r2) - n;
	double klog = log2((eqSize * eqSize * eqSize) / double(r * n * n * 2));
	t0 = t0 + klog;

	double t1 = 1.86*t;

	double t2 = 1.86 * (m * r2+e);

	double t3 = 1.86 * m * r2 + 3 * m * r1 - n - 1.14*(t+e);

	double t4 = t3;
	if (t2 > t3)
		t4 = t2;

	cout << t0 << " " << t1 << " " << t4 << endl;
}

void testSuccessRate() {
	LowMC lowmc;
	bool p0[N], k[N], c0[N], p1[N], c1[N];
	vector<vector<bool> > SOut0, SOut1, LOut0, LOut1;
	SOut0.resize(ATTACK_ROUNDS), SOut1.resize(ATTACK_ROUNDS), LOut0.resize(ATTACK_ROUNDS), LOut1.resize(ATTACK_ROUNDS);
	for (int i = 0; i < ATTACK_ROUNDS; i++) {
		SOut0[i].resize(N), SOut1[i].resize(N), LOut0[i].resize(N), LOut1[i].resize(N);
	}
	vector<vector<bool> > SOutDiff, LOutDiff;
	SOutDiff.resize(ATTACK_ROUNDS), LOutDiff.resize(ATTACK_ROUNDS);
	for (int i = 0; i < ATTACK_ROUNDS; i++) {
		SOutDiff[i].resize(N), LOutDiff[i].resize(N);
	}

	srand(time(NULL));
	int success = 0;
	int recovered = 0;
	int recoveryAttempts = 0;
	int CNT = 500;
	const int PAIRS_PER_TRIAL = 4;
	const int TOTAL_PAIRS = CNT * PAIRS_PER_TRIAL;

	std::mt19937 mt;         
	std::random_device rnd; 
	mt.seed(rnd());

	bool diff[N], internalDiff[N];
	lowmc.findInputDifference(internalDiff, diff);

	for (int trial = 0; trial < CNT; trial++) {
		for (int i = 0; i < N / 16; i++) {
			unsigned int v1 = mt();
			for (int j = 0; j < 16; j++) {
				k[i * 16 + j] = (v1 >> j) & 0x1;
			}
		}

		for (int pairIndex = 0; pairIndex < PAIRS_PER_TRIAL; pairIndex++) {
			for (int i = 0; i < N / 16; i++) {
				unsigned int v0 = mt();
				for (int j = 0; j < 16; j++) {
					p0[i * 16 + j] = (v0 >> j) & 0x1;
				}
			}

			for (int i = 0; i < N; i++) {
				p1[i] = p0[i] ^ diff[i];
			}

			lowmc.encryptionDetails(p0, k, c0, ATTACK_ROUNDS, LOut0, SOut0);
			lowmc.encryptionDetails(p1, k, c1, ATTACK_ROUNDS, LOut1, SOut1);

			int activeCnt = 0;
			int checkedRounds = ATTACK_R1 + ATTACK_R2;
			for (int i = ATTACK_ROUNDS; i > ATTACK_ROUNDS - checkedRounds; i--) {
				if (SOut0[i - 1][0] != SOut1[i - 1][0]
					|| SOut0[i - 1][1] != SOut1[i - 1][1]
					|| SOut0[i - 1][2] != SOut1[i - 1][2]) {
					activeCnt++;
				}
			}
			if (activeCnt >= 33) {
				success++;
			}

			if (pairIndex == 0 && activeCnt >= 33) {
				recoveryAttempts++;
				for (int i = 0; i < ATTACK_ROUNDS; i++) {
					for (int j = 0; j < N; j++) {
						SOutDiff[i][j] = SOut0[i][j] ^ SOut1[i][j];
						LOutDiff[i][j] = LOut0[i][j] ^ LOut1[i][j];
					}
				}

				int status = lowmc.keyRecovery(k, c0, ATTACK_ROUNDS, LOut0, SOut0, SOutDiff, LOutDiff);
				if (status > 0) {
					recovered += PAIRS_PER_TRIAL;
				}
			}
		}
	}

	cout << "rate:" << (double)(success) / TOTAL_PAIRS << endl;
	cout << "key recovery success rate over all pairs:" << (double)(recovered) / TOTAL_PAIRS << endl;

	for (int i = 0; i < ATTACK_ROUNDS; i++) {
		SOut0[i].clear(), SOut1[i].clear(), LOut0[i].clear(), LOut1[i].clear();
		SOutDiff[i].clear(), LOutDiff[i].clear();
	}
	SOut0.clear(), SOut1.clear(), LOut0.clear(), LOut1.clear();
	SOutDiff.clear(), LOutDiff.clear();
}

void findInputDiff() {
	LowMC lowmc;
	bool x[N], inputDiff[N];
	for (int i = 0; i < N; i++) {
		x[i] = 1;
		inputDiff[i] = 0;
	}
	lowmc.findInputDifference(x, inputDiff);
}

void computeDh() {
	LowMC lowmc;
	bool x[N], inputDiff[N];
	lowmc.findInputDifference(x, inputDiff);
	lowmc.enumerateDiffForwards(x);
}

void online(int pairNum) {
	if (pairNum < 1) {
		pairNum = 1;
	}
	//offline
	LowMC lowmc;
	bool x0[N], inputDiff[N];
	lowmc.findInputDifference(x0, inputDiff);
	lowmc.constructPQ(x0);
	//online

	bool p0[N], p1[N], key[N], c0[N], c1[N], x1[N];
	vector<vector<bool> > SOut0, SOut1, LOut0, LOut1;
	SOut0.resize(ATTACK_ROUNDS), SOut1.resize(ATTACK_ROUNDS);
	LOut0.resize(ATTACK_ROUNDS), LOut1.resize(ATTACK_ROUNDS);
	for (int i = 0; i < ATTACK_ROUNDS; i++) {
		SOut0[i].resize(N), SOut1[i].resize(N);
		LOut0[i].resize(N), LOut1[i].resize(N);
	}

	std::mt19937 mt;
	std::random_device rnd;
	mt.seed(rnd());
	for (int i = 0; i < N / 16; i++) {
		unsigned int v1 = mt();
		for (int j = 0; j < 16; j++) {
			key[i * 16 + j] = (v1 >> j) & 0x1;
		}
	}

	unsigned long long totalCnt[3] = { 0,0,0 };
	vector<AttackRecord> records;
	for (int pairIndex = 0; pairIndex < pairNum; pairIndex++) {
		for (int i = 0; i < N / 16; i++) {
			unsigned int v0 = mt();
			for (int j = 0; j < 16; j++) {
				p0[i * 16 + j] = (v0 >> j) & 0x1;
			}
		}
		for (int i = 0; i < N; i++) {
			p1[i] = p0[i] ^ inputDiff[i];
		}

		lowmc.encryptionDetails(p0, key, c0, ATTACK_ROUNDS, LOut0, SOut0);
		lowmc.encryptionDetails(p1, key, c1, ATTACK_ROUNDS, LOut1, SOut1);
		for (int i = 0; i < N; i++) {
			x1[i] = c0[i] ^ c1[i];
		}

		cout << "ciphertext difference " << pairIndex + 1 << " for backward enumeration:" << endl;
		for (int i = 0; i < N; i++) {
			cout << x1[i] << ",";
		}
		cout << endl;

		unsigned long long cnt[3] = { 0,0,0 };
		AttackRecord baseRecord;
		baseRecord.p0 = arrayToVector(p0, N);
		baseRecord.p1 = arrayToVector(p1, N);
		baseRecord.c0 = arrayToVector(c0, N);
		baseRecord.c1 = arrayToVector(c1, N);
		baseRecord.key = arrayToVector(key, N);
		baseRecord.SOut0 = SOut0;
		baseRecord.LOut0 = LOut0;
		lowmc.enumerateDiffBackwards(x1, cnt, records, baseRecord);
		totalCnt[0] += cnt[0];
		totalCnt[1] += cnt[1];
		totalCnt[2] += cnt[2];
	}

	cout << dec << "DDT-valid matched differences: " << totalCnt[2] << endl;
	cout << dec << "time complexity: " << totalCnt[1] << endl;
	cout << dec << "plaintext pairs: " << pairNum << endl;
	saveCommand2Records(records);
	cout << dec << "saved matched paths: " << records.size() << " -> " << COMMAND2_PATH_FILE << endl;
	lowmc.clearDifferenceTable();
}

void recoverFromCommand2Paths() {
	vector<AttackRecord> records;
	if (!loadCommand2Records(records)) {
		cout << "failed to load " << COMMAND2_PATH_FILE << endl;
		cout << "please run command 2 first." << endl;
		return;
	}

	LowMC lowmc;
	int recovered = 0;
	for (int i = 0; i < records.size(); i++) {
		bool recoveredKey[N], p0[N], p1[N], c0[N], c1[N], testC0[N], testC1[N];
		vectorToArray(records[i].p0, p0);
		vectorToArray(records[i].p1, p1);
		vectorToArray(records[i].c0, c0);
		vectorToArray(records[i].c1, c1);
		int status = lowmc.keyRecoveryAttack(p0, p1, c0, c1, ATTACK_ROUNDS, records[i].SOutDiff, records[i].LOutDiff, recoveredKey);

		bool verified = false;
		if (status > 0) {
			vector<vector<bool> > SOut0(ATTACK_ROUNDS, vector<bool>(N, 0));
			vector<vector<bool> > SOut1(ATTACK_ROUNDS, vector<bool>(N, 0));
			vector<vector<bool> > LOut0(ATTACK_ROUNDS, vector<bool>(N, 0));
			vector<vector<bool> > LOut1(ATTACK_ROUNDS, vector<bool>(N, 0));
			lowmc.encryptionDetails(p0, recoveredKey, testC0, ATTACK_ROUNDS, LOut0, SOut0);
			lowmc.encryptionDetails(p1, recoveredKey, testC1, ATTACK_ROUNDS, LOut1, SOut1);
			verified = true;
			for (int j = 0; j < N; j++) {
				if (testC0[j] != c0[j] || testC1[j] != c1[j]) {
					verified = false;
					break;
				}
			}
		}

		cout << "path " << (i + 1) << "/" << records.size() << " keyRecoveryAttack status: " << status << ", verified: " << verified << endl;
		if (verified) {
			recovered++;
		}
	}
	cout << "verified recovered keys: " << recovered << "/" << records.size() << endl;
}

void diagnoseCommand2Paths() {
	vector<AttackRecord> records;
	if (!loadCommand2Records(records) || records.empty()) {
		cout << "failed to load " << COMMAND2_PATH_FILE << endl;
		return;
	}

	LowMC lowmc;
	bool p0[N], p1[N], key[N], c0[N], c1[N];
	vectorToArray(records[0].p0, p0);
	vectorToArray(records[0].p1, p1);
	vectorToArray(records[0].key, key);
	vector<vector<bool> > SOut0(ATTACK_ROUNDS, vector<bool>(N, 0));
	vector<vector<bool> > SOut1(ATTACK_ROUNDS, vector<bool>(N, 0));
	vector<vector<bool> > LOut0(ATTACK_ROUNDS, vector<bool>(N, 0));
	vector<vector<bool> > LOut1(ATTACK_ROUNDS, vector<bool>(N, 0));
	lowmc.encryptionDetails(p0, key, c0, ATTACK_ROUNDS, LOut0, SOut0);
	lowmc.encryptionDetails(p1, key, c1, ATTACK_ROUNDS, LOut1, SOut1);
	bool firstAllMatch = true;
	for (int r = ATTACK_R0; r < ATTACK_ROUNDS; r++) {
		bool sMatch = true;
		bool lMatch = true;
		for (int j = 0; j < N; j++) {
			if ((SOut0[r][j] ^ SOut1[r][j]) != records[0].SOutDiff[r][j]) {
				sMatch = false;
			}
			if ((LOut0[r][j] ^ LOut1[r][j]) != records[0].LOutDiff[r][j]) {
				lMatch = false;
			}
		}
		if (!sMatch || !lMatch) {
			cout << "first saved path differs from the real encrypted path at round " << r << endl;
			cout << "SOutDiff match: " << sMatch << ", LOutDiff match: " << lMatch << endl;
			firstAllMatch = false;
			break;
		}
	}
	if (firstAllMatch) {
		cout << "first saved path matches the real encrypted path for checked rounds." << endl;
	}

	int exactMatches = 0;
	for (int i = 0; i < records.size(); i++) {
		bool allMatch = true;
		for (int r = ATTACK_R0; r < ATTACK_ROUNDS && allMatch; r++) {
			for (int j = 0; j < N; j++) {
				if ((SOut0[r][j] ^ SOut1[r][j]) != records[i].SOutDiff[r][j]) {
					allMatch = false;
					break;
				}
				if ((LOut0[r][j] ^ LOut1[r][j]) != records[i].LOutDiff[r][j]) {
					allMatch = false;
					break;
				}
			}
		}
		if (allMatch) {
			exactMatches++;
		}
	}
	cout << "saved paths matching the real encrypted path: " << exactMatches << "/" << records.size() << endl;
}

int main() {
	int cmd = 0;
	while (1) {
		cout << endl;
		cout << "*******************************************************************" << endl;
		cout << "*Notice 1: If you input 1, the file uvalue.txt will be regenerated." << endl;
		cout << "*However, when you input 2, the file uvalue.txt is needed." << endl;
		cout << "*So, if you first input 1, wait unti command-1 is finished before doing other experiments" << endl;
		cout << "*******************************************************************" << endl;
		cout << "0 -> exit" << endl;
		cout << "1 -> precomputation (contruct tables)" << endl;
		cout << "2 -> check the difference enumeration via MITM algebraic technique" << endl;
		cout << "3 -> key recovery with paths saved by command 2" << endl;
		cout << "input cmd(0/1/2/3/):";
		cin >> cmd;
		if (cmd == 1) {
			computeDh();
		}
		else if (cmd == 2) {
			int pairNum = 4;
			cout << "input plaintext pair count(use 1 for single pair):";
			cin >> pairNum;
			online(pairNum);
		}
		else if (cmd == 3) {
			recoverFromCommand2Paths();
		}
		else
			break;
	}

	return 0;
}
