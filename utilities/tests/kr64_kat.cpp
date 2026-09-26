// Known-answer test for src/KeyedRNG.H against the Python reference (popgen/kr64.py).
//
// Build and run from the repository root:
//   g++ -std=c++17 -O2 -ffp-contract=off -Isrc utilities/tests/kr64_kat.cpp -o /tmp/kr64_kat
//   /tmp/kr64_kat utilities/UrbanPop-scripts/popgen/kr64_vectors.txt \
//                 utilities/UrbanPop-scripts/popgen/kr64_cdf_vectors.txt
//
// Each vector line is `n w1..wn key d0 d1 d2 u01_bits index7 index1000003 index2^40`, written by
// `python -m popgen.stages`. The optional CDF file holds `kind m w1..wm j result` lines (kind 0 =
// intCdf with integer weights, 1 = floatCdf with hex-float weights), each drawn with
// draw(key(99, 0, j), 0). Exits non-zero on the first mismatch.

#include "KeyedRNG.H"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int checkCdf (const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return -1;
    }
    std::string line;
    int n_ok = 0;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        std::istringstream ss(line);
        int kind, m;
        ss >> kind >> m;
        std::vector<std::int64_t> icum(m);
        std::vector<double> fcum(m);
        std::int64_t isum = 0;
        double fsum = 0.0;
        for (int i = 0; i < m; i++) {
            std::string tok;
            ss >> tok;
            if (kind == 0) {
                isum += std::strtoll(tok.c_str(), nullptr, 10);
                icum[i] = isum;
            } else {
                fsum += std::strtod(tok.c_str(), nullptr);
                fcum[i] = fsum;
            }
        }
        std::int64_t j, want;
        ss >> j >> want;
        std::uint64_t x = PopGen::KR64(99, 0, j).u64(0);
        std::int64_t got = kind == 0 ? PopGen::intCdf(icum.data(), m, x) : PopGen::floatCdf(fcum.data(), m, x);
        if (got != want) {
            std::cerr << "CDF MISMATCH (got " << got << "): " << line << "\n";
            return -1;
        }
        n_ok++;
    }
    return n_ok;
}

int main (int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: kr64_kat kr64_vectors.txt [kr64_cdf_vectors.txt]\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }
    std::string line;
    int n_ok = 0, lineno = 0;
    while (std::getline(in, line)) {
        lineno++;
        if (line.empty()) { continue; }
        std::istringstream ss(line);
        int n;
        ss >> n;
        std::vector<std::int64_t> w(n);
        for (auto& x : w) {
            ss >> x;
        }
        std::uint64_t key, d[3], bits;
        std::int64_t idx[3];
        ss >> key >> d[0] >> d[1] >> d[2] >> bits >> idx[0] >> idx[1] >> idx[2];

        std::uint64_t h = PopGen::KR64Const::IV;
        for (auto x : w) {
            h = PopGen::absorb(h, x);
        }
        PopGen::KR64 k;
        k.h = h;
        double u = k.u01(0);
        std::uint64_t ubits;
        std::memcpy(&ubits, &u, sizeof(u));
        const std::int64_t ns[3] = {7, 1000003, std::int64_t(1) << 40};
        bool ok = (h == key) && (ubits == bits);
        for (int j = 0; j < 3; j++) {
            ok = ok && (k.u64(j) == d[j]) && (k.index(1, ns[j]) == idx[j]);
        }
        if (n >= 3) {
            // The (seed, rep, stage) constructor must agree with folding the words one by one.
            PopGen::KR64 k2(w[0], w[1], w[2]);
            for (int i = 3; i < n; i++) {
                k2 = k2.with(w[i]);
            }
            ok = ok && (k2.h == key);
        }
        if (!ok) {
            std::cerr << "MISMATCH on line " << lineno << ": " << line << "\n";
            return 1;
        }
        n_ok++;
    }
    std::cout << "kr64: " << n_ok << " known-answer vectors match\n";
    if (argc == 3) {
        int n_cdf = checkCdf(argv[2]);
        if (n_cdf <= 0) { return 1; }
        std::cout << "kr64: " << n_cdf << " CDF vectors match\n";
    }
    return n_ok > 0 ? 0 : 1;
}
