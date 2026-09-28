/*! @file PopulationBundle.H
    \brief Reader for the population-generation bundle (UPPB) written by build_precompute.py.

    A C++ port of utilities/UrbanPop-scripts/popgen/bundle.py. The bundle holds everything
    population generation needs and nothing random: per-PUMA P-MEDM solve inputs, donor
    households and persons, LODES flows, schools, CBP tables, county adjacency. Layout,
    little-endian:

        header      "<4I Q": magic "UPPB", format version, section count, 0, directory offset
        sections    each deflate-compressed (codec 1, a zlib stream) or raw (codec 0), back to back
        directory   per section: u16 name length + name, u16 dtype length + numpy dtype.str,
                    u8 ndim, u64 per dimension, u32 codec, u64 offset, u64 stored bytes,
                    u64 raw bytes

    Every section is inflated at open into its own 8-byte-aligned buffer and is then read through
    get<T>(), which checks that T matches the stored numpy dtype, so a section whose type changes
    in the writer fails loudly instead of being reinterpreted. Plain C++17 plus zlib and no AMReX,
    so utilities/tests/bundle_read.cpp can check it against the Python reader standalone; errors
    are std::runtime_error for the caller to turn into amrex::Abort.
*/
#ifndef POPULATION_BUNDLE_H_
#define POPULATION_BUNDLE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace PopGen {

/*! numpy dtype.str of each element type a section can hold. */
template <class T>
struct NumpyDtype;
template <>
struct NumpyDtype<std::int64_t> {
    static constexpr const char* str = "<i8";
};
template <>
struct NumpyDtype<std::int32_t> {
    static constexpr const char* str = "<i4";
};
template <>
struct NumpyDtype<std::int16_t> {
    static constexpr const char* str = "<i2";
};
template <>
struct NumpyDtype<std::int8_t> {
    static constexpr const char* str = "|i1";
};
template <>
struct NumpyDtype<std::uint64_t> {
    static constexpr const char* str = "<u8";
};
template <>
struct NumpyDtype<std::uint32_t> {
    static constexpr const char* str = "<u4";
};
template <>
struct NumpyDtype<std::uint16_t> {
    static constexpr const char* str = "<u2";
};
template <>
struct NumpyDtype<std::uint8_t> {
    static constexpr const char* str = "|u1";
};
template <>
struct NumpyDtype<double> {
    static constexpr const char* str = "<f8";
};
template <>
struct NumpyDtype<float> {
    static constexpr const char* str = "<f4";
};

/*! Read-only view of one section's elements, row-major for multi-dimensional sections. */
template <class T>
struct ArrayView {
    const T* ptr = nullptr;
    std::size_t n = 0;
    std::vector<std::uint64_t> shape;

    std::size_t size () const noexcept { return n; }
    const T* data () const noexcept { return ptr; }
    const T* begin () const noexcept { return ptr; }
    const T* end () const noexcept { return ptr + n; }
    const T& operator[](std::size_t i) const noexcept { return ptr[i]; }
    /*! Element (i, j) of a 2-D section. */
    const T& operator()(std::size_t i, std::size_t j) const noexcept { return ptr[i * shape[1] + j]; }
};

struct BundleSection {
    std::string dtype;                  /*!< numpy dtype.str, e.g. "<i8" */
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> storage; /*!< raw bytes, in 8-byte words so any element type is aligned */
    std::size_t nbytes = 0;

    const char* bytes () const noexcept { return reinterpret_cast<const char*>(storage.data()); }
};

class PopulationBundle {
  public:
    static constexpr std::uint32_t MAGIC = 0x42505055; // "UPPB"
    static constexpr std::uint32_t FORMAT_VERSION = 4;

    PopulationBundle() = default;

    /*! Read and inflate every section; throws std::runtime_error on a malformed file or a
        format version other than FORMAT_VERSION. */
    explicit PopulationBundle(const std::string& path);

    std::uint32_t version () const noexcept { return m_version; }
    bool has (const std::string& name) const { return m_sections.count(name) != 0; }
    const BundleSection& section(const std::string& name) const;
    const std::map<std::string, BundleSection>& sections () const noexcept { return m_sections; }

    /*! Typed view of a section; throws if it is missing or stored with another dtype. */
    template <class T>
    ArrayView<T> get (const std::string& name) const {
        const BundleSection& s = section(name);
        if (s.dtype != NumpyDtype<T>::str) {
            throw std::runtime_error("bundle section " + name + " is " + s.dtype + ", read as " + NumpyDtype<T>::str);
        }
        ArrayView<T> v;
        v.ptr = reinterpret_cast<const T*>(s.bytes());
        v.n = s.nbytes / sizeof(T);
        v.shape = s.shape;
        return v;
    }

    /*! A string list stored as name.blob (concatenated UTF-8) + name.offsets (start offsets). */
    std::vector<std::string> strings(const std::string& name) const;

    /*! The bundle's metadata, a JSON document (section "meta"). */
    std::string meta() const;

  private:
    std::uint32_t m_version = 0;
    std::map<std::string, BundleSection> m_sections;
};

/*! zlib CRC-32 of a section's raw bytes; the checksum utilities/tests/bundle_read.cpp compares
    against `python -m popgen.bundle --manifest` (Python's zlib.crc32). */
std::uint32_t sectionCrc32(const BundleSection& s) noexcept;

} // namespace PopGen

#endif
