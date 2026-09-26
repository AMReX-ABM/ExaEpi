/*! @file Sha256.H
    \brief SHA-256 (FIPS 180-4), for population digests.

    A generated population is checked stage by stage against the Python reference generator by
    digest: the first 16 hex digits of SHA-256 over int64 copies of each stage's output arrays
    (generate_exaepi.py, digest). Header-only and dependency-free.
*/
#ifndef EXAEPI_SHA256_H_
#define EXAEPI_SHA256_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace PopGen {

class Sha256 {
  public:
    Sha256 () { reset(); }

    void reset () {
        static constexpr std::uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(m_h, init, sizeof(m_h));
        m_len = 0;
        m_fill = 0;
    }

    void update (const void* data, std::size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        m_len += n;
        while (n > 0) {
            const std::size_t take = (64 - m_fill < n) ? 64 - m_fill : n;
            std::memcpy(m_block + m_fill, p, take);
            m_fill += take;
            p += take;
            n -= take;
            if (m_fill == 64) {
                compress(m_block);
                m_fill = 0;
            }
        }
    }

    /*! Lowercase hex digest; the object must be reset() before reuse. */
    std::string hex () {
        const std::uint64_t bits = m_len * 8;
        const unsigned char pad = 0x80;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (m_fill != 56) {
            update(&zero, 1);
        }
        unsigned char be[8];
        for (int i = 0; i < 8; ++i) {
            be[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        }
        update(be, 8);
        static const char* digits = "0123456789abcdef";
        std::string out;
        for (std::uint32_t w : m_h) {
            for (int s = 28; s >= 0; s -= 4) {
                out += digits[(w >> s) & 0xF];
            }
        }
        return out;
    }

  private:
    static std::uint32_t rotr (std::uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

    void compress (const unsigned char* b) {
        static constexpr std::uint32_t k[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                                                0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                                                0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                                                0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                                                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                                                0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                                                0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                                                0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                                                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                                                0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                                                0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t(b[4 * i]) << 24) | (std::uint32_t(b[4 * i + 1]) << 16) | (std::uint32_t(b[4 * i + 2]) << 8) |
                   std::uint32_t(b[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = m_h[0], bb = m_h[1], c = m_h[2], d = m_h[3], e = m_h[4], f = m_h[5], g = m_h[6], h = m_h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + S1 + ch + k[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & bb) ^ (a & c) ^ (bb & c);
            const std::uint32_t t2 = S0 + mj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = bb;
            bb = a;
            a = t1 + t2;
        }
        m_h[0] += a;
        m_h[1] += bb;
        m_h[2] += c;
        m_h[3] += d;
        m_h[4] += e;
        m_h[5] += f;
        m_h[6] += g;
        m_h[7] += h;
    }

    std::uint32_t m_h[8];
    unsigned char m_block[64];
    std::size_t m_fill;
    std::uint64_t m_len;
};

/*! Accumulates arrays as int64 (little-endian, as on every supported host) and returns the
    16-hex-digit digest generate_exaepi.py's digest() computes over the same arrays. */
class Digest {
  public:
    template <class T>
    Digest& add (const std::vector<T>& v) {
        return add(v.data(), v.size());
    }
    template <class T>
    Digest& add (const T* p, std::size_t n) {
        std::int64_t buf[512];
        std::size_t i = 0;
        while (i < n) {
            const std::size_t m = (n - i < 512) ? n - i : 512;
            for (std::size_t j = 0; j < m; ++j) {
                buf[j] = static_cast<std::int64_t>(p[i + j]);
            }
            m_sha.update(buf, m * sizeof(std::int64_t));
            i += m;
        }
        return *this;
    }
    std::string hex16 () { return m_sha.hex().substr(0, 16); }

  private:
    Sha256 m_sha;
};

} // namespace PopGen

#endif
