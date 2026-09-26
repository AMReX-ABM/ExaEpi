// Check src/PopulationBundle against the Python reader (popgen/bundle.py).
//
// Build and run from the repository root:
//   g++ -std=c++17 -O2 -Isrc utilities/tests/bundle_read.cpp src/PopulationBundle.cpp -lz -o /tmp/bundle_read
//   /tmp/bundle_read BUNDLE > cpp.txt
//   (cd utilities/UrbanPop-scripts && python -m popgen.bundle --manifest BUNDLE) > py.txt
//   diff py.txt cpp.txt
//
// Prints one line per section -- name, numpy dtype, shape, raw bytes, CRC-32 -- in the format of
// `python -m popgen.bundle --manifest`, so the two must be identical. Also checks that a typed read
// with the wrong element type is refused, and that string lists decode. Exits non-zero on failure.

#include "PopulationBundle.H"

#include <cstdio>
#include <iostream>
#include <string>

int main (int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: bundle_read BUNDLE\n";
        return 2;
    }
    try {
        const PopGen::PopulationBundle b(argv[1]);
        for (const auto& [name, s] : b.sections()) {
            std::string shape;
            for (std::size_t i = 0; i < s.shape.size(); ++i) {
                shape += (i ? "x" : "") + std::to_string(s.shape[i]);
            }
            std::printf("%s %s %s %zu %08x\n", name.c_str(), s.dtype.c_str(), shape.c_str(), s.nbytes, PopGen::sectionCrc32(s));
        }

        bool refused = false;
        try {
            b.get<std::int64_t>("solve.est_bg"); // stored as float64
        } catch (const std::runtime_error&) { refused = true; }
        if (!refused) {
            std::cerr << "FAIL: get<int64_t> on a float64 section was not refused\n";
            return 1;
        }
        const auto pumas = b.strings("solve.puma");
        const auto est = b.get<double>("solve.est_bg");
        if (pumas.empty() || est.shape.size() != 2 || est.size() != est.shape[0] * est.shape[1]) {
            std::cerr << "FAIL: string list or 2-D section shape\n";
            return 1;
        }
        std::cerr << "bundle_read: " << b.sections().size() << " sections, " << pumas.size() << " PUMAs (first " << pumas.front()
                  << "), est_bg " << est.shape[0] << "x" << est.shape[1] << "\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
