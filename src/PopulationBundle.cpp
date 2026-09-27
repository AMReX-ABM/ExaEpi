/*! @file PopulationBundle.cpp
    \brief Reader for the population-generation bundle; see PopulationBundle.H.
*/
#include "PopulationBundle.H"

#include <fstream>
#include <zlib.h>

namespace PopGen {

namespace {

template <class T>
T readScalar (std::ifstream& f, const std::string& path) {
    T v;
    if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) { throw std::runtime_error(path + ": truncated bundle"); }
    return v;
}

std::string readString (std::ifstream& f, const std::string& path) {
    const auto len = readScalar<std::uint16_t>(f, path);
    std::string s(len, '\0');
    if (len > 0 && !f.read(&s[0], len)) { throw std::runtime_error(path + ": truncated bundle"); }
    return s;
}

struct Entry {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::uint32_t codec;
    std::uint64_t offset, nstored, nraw;
};

constexpr std::uint32_t CODEC_RAW = 0;
constexpr std::uint32_t CODEC_DEFLATE = 1;

} // namespace

PopulationBundle::PopulationBundle (const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open population bundle " + path); }
    const auto magic = readScalar<std::uint32_t>(f, path);
    m_version = readScalar<std::uint32_t>(f, path);
    const auto n_sections = readScalar<std::uint32_t>(f, path);
    readScalar<std::uint32_t>(f, path); // reserved
    const auto dir_offset = readScalar<std::uint64_t>(f, path);
    if (magic != MAGIC) {
        // Large bundles are kept in git-lfs; a clone that has not fetched them holds a small text
        // pointer in their place.
        const std::string lfs = "version https://git-lfs";
        std::string head(lfs.size(), '\0');
        f.clear();
        f.seekg(0);
        f.read(&head[0], static_cast<std::streamsize>(head.size()));
        if (head == lfs) {
            throw std::runtime_error(path + " is a git-lfs pointer, not the bundle: run `git lfs pull` to fetch it");
        }
        throw std::runtime_error(path + ": not a population bundle (bad magic)");
    }
    if (m_version != FORMAT_VERSION) {
        throw std::runtime_error(path + ": bundle format " + std::to_string(m_version) + ", this build reads " +
                                 std::to_string(FORMAT_VERSION) + " -- rebuild it with build_precompute.py");
    }

    f.seekg(static_cast<std::streamoff>(dir_offset));
    std::vector<Entry> entries(n_sections);
    for (auto& e : entries) {
        e.name = readString(f, path);
        e.dtype = readString(f, path);
        const auto ndim = readScalar<std::uint8_t>(f, path);
        e.shape.resize(ndim);
        for (auto& d : e.shape) {
            d = readScalar<std::uint64_t>(f, path);
        }
        e.codec = readScalar<std::uint32_t>(f, path);
        e.offset = readScalar<std::uint64_t>(f, path);
        e.nstored = readScalar<std::uint64_t>(f, path);
        e.nraw = readScalar<std::uint64_t>(f, path);
    }

    std::vector<char> stored;
    for (const auto& e : entries) {
        BundleSection s;
        s.dtype = e.dtype;
        s.shape = e.shape;
        s.nbytes = e.nraw;
        s.storage.assign((e.nraw + 7) / 8, 0);
        char* out = reinterpret_cast<char*>(s.storage.data());
        f.seekg(static_cast<std::streamoff>(e.offset));
        if (e.codec == CODEC_RAW) {
            if (e.nstored != e.nraw) { throw std::runtime_error(path + ": section " + e.name + " size mismatch"); }
            if (e.nraw > 0 && !f.read(out, static_cast<std::streamsize>(e.nraw))) {
                throw std::runtime_error(path + ": truncated section " + e.name);
            }
        } else if (e.codec == CODEC_DEFLATE) {
            stored.resize(e.nstored);
            if (!f.read(stored.data(), static_cast<std::streamsize>(e.nstored))) {
                throw std::runtime_error(path + ": truncated section " + e.name);
            }
            uLongf len = static_cast<uLongf>(e.nraw);
            const int rc = uncompress(reinterpret_cast<Bytef*>(out), &len, reinterpret_cast<const Bytef*>(stored.data()),
                                      static_cast<uLong>(e.nstored));
            if (rc != Z_OK || len != e.nraw) {
                throw std::runtime_error(path + ": cannot inflate section " + e.name + " (zlib " + std::to_string(rc) + ")");
            }
        } else {
            throw std::runtime_error(path + ": section " + e.name + " uses unknown codec " + std::to_string(e.codec));
        }
        m_sections.emplace(e.name, std::move(s));
    }
}

const BundleSection& PopulationBundle::section (const std::string& name) const {
    auto it = m_sections.find(name);
    if (it == m_sections.end()) { throw std::runtime_error("population bundle has no section " + name); }
    return it->second;
}

std::vector<std::string> PopulationBundle::strings (const std::string& name) const {
    const BundleSection& blob = section(name + ".blob");
    const auto off = get<std::int64_t>(name + ".offsets");
    std::vector<std::string> out;
    if (off.size() == 0) { return out; }
    out.reserve(off.size() - 1);
    for (std::size_t i = 0; i + 1 < off.size(); ++i) {
        if (off[i] < 0 || off[i + 1] < off[i] || static_cast<std::size_t>(off[i + 1]) > blob.nbytes) {
            throw std::runtime_error("population bundle string list " + name + " has bad offsets");
        }
        out.emplace_back(blob.bytes() + off[i], static_cast<std::size_t>(off[i + 1] - off[i]));
    }
    return out;
}

std::string PopulationBundle::meta () const {
    const BundleSection& s = section("meta");
    return std::string(s.bytes(), s.nbytes);
}

std::uint32_t sectionCrc32 (const BundleSection& s) noexcept {
    uLong crc = crc32(0L, Z_NULL, 0);
    const auto* p = reinterpret_cast<const Bytef*>(s.bytes());
    std::size_t left = s.nbytes;
    while (left > 0) { // crc32 takes a uInt length
        const auto chunk = static_cast<uInt>(left < (1u << 30) ? left : (1u << 30));
        crc = crc32(crc, p, chunk);
        p += chunk;
        left -= chunk;
    }
    return static_cast<std::uint32_t>(crc);
}

} // namespace PopGen
