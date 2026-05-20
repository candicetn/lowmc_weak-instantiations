# LowMC Weak Instantiation Attack Verification

This repository contains code and experimental artifacts for verifying algebraic meet-in-the-middle attacks on a concrete weak random linear-layer instantiation of LowMC.



To compile the attack code, use:

```bash
g++ LowMC.cpp main.cpp -O3 -std=c++11 -o lowmc_attack
```