#pragma once

#include <hdf5.h>
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace VVM::IO {

struct Hdf5CompressionOptions {
    bool enabled = false;
    int level = 1;
    std::size_t minimum_bytes = 1024 * 1024;
};

// Native HDF5 is used only for the optional compressed sink. ADIOS2 still
// carries SST data and defines the field schema. Every rank must call write()
// in the same order, including ranks whose selection is empty.
class CompressedHdf5Writer {
    class Handle {
        hid_t id_;
        herr_t (*close_)(hid_t);
    public:
        Handle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {
            if (id < 0) throw std::runtime_error("HDF5 object creation failed");
        }
        ~Handle() { if (id_ >= 0) close_(id_); }
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        operator hid_t() const { return id_; }
        void close() {
            if (id_ >= 0) {
                const auto id = id_;
                id_ = -1;
                check(close_(id), "close");
            }
        }
    };

    static void check(herr_t status, const char* operation) {
        if (status < 0) throw std::runtime_error(std::string("Compressed HDF5: ") + operation);
    }
    static hid_t create_file(const std::string& filename, MPI_Comm comm,
        const Hdf5CompressionOptions& options) {
        if (options.level < 0 || options.level > 9 || options.minimum_bytes == 0)
            throw std::invalid_argument("Invalid HDF5 compression level or minimum_bytes");
        // Open MPI 4.1.7's OMPIO stalls on successive filtered writes on this
        // stack. submit.py selects ROMIO before MPI starts. Fail clearly for
        // manual multi-rank launches instead of hanging during output.
        char library[MPI_MAX_LIBRARY_VERSION_STRING];
        int length = 0, ranks = 1;
        MPI_Get_library_version(library, &length);
        MPI_Comm_size(comm, &ranks);
        const char* io = std::getenv("OMPI_MCA_io");
        if (ranks > 1 && std::string(library, length).find("Open MPI") != std::string::npos &&
            (!io || std::string(io) != "romio321"))
            throw std::runtime_error("Compressed HDF5 with Open MPI requires OMPI_MCA_io=romio321 "
                "(or mpirun --mca io romio321); submit.py sets this automatically");
        for (const auto filter : {H5Z_FILTER_SHUFFLE, H5Z_FILTER_DEFLATE, H5Z_FILTER_FLETCHER32}) {
            unsigned flags = 0;
            if (H5Zfilter_avail(filter) <= 0 || H5Zget_filter_info(filter, &flags) < 0 ||
                !(flags & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
                !(flags & H5Z_FILTER_CONFIG_DECODE_ENABLED))
                throw std::runtime_error("HDF5 shuffle/deflate/Fletcher32 unavailable");
        }
        Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
        check(H5Pset_fapl_mpio(access, comm, MPI_INFO_NULL), "set MPI file access");
        check(H5Pset_all_coll_metadata_ops(access, 1), "collective metadata reads");
        check(H5Pset_coll_metadata_write(access, 1), "collective metadata writes");
        return H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access);
    }
    template<class T> static hid_t type() {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                      "Unsupported HDF5 output type");
        if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
        else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
        else return H5T_NATIVE_UINT64;
    }
    Handle file_;
    Handle group_;
    Hdf5CompressionOptions options_;
    int rank_ = 0;

public:
    CompressedHdf5Writer(const std::string& filename, MPI_Comm comm,
        Hdf5CompressionOptions options)
        : file_(create_file(filename, comm, options), H5Fclose),
          group_(H5Gcreate2(file_, "Step0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose),
          options_(options) {
        MPI_Comm_rank(comm, &rank_);
        attribute<uint64_t>("NumSteps", {1});
    }

    template<class T>
    void write(const std::string& name, const std::vector<size_t>& shape,
        const std::vector<size_t>& start, const std::vector<size_t>& count, const T* data) {
        if (shape.size() != start.size() || shape.size() != count.size())
            throw std::invalid_argument("HDF5 shape/selection rank mismatch: " + name);
        bool empty = false;
        std::size_t bytes = sizeof(T);
        for (size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] == 0 || start[d] > shape[d] || count[d] > shape[d] - start[d])
                throw std::invalid_argument("Invalid HDF5 selection: " + name);
            empty = empty || count[d] == 0;
            if (bytes > std::numeric_limits<size_t>::max() / shape[d])
                throw std::overflow_error("HDF5 dataset size overflow");
            bytes *= shape[d];
        }
        if (shape.empty()) empty = rank_ != 0;
        if (!empty && !data) throw std::invalid_argument("Missing HDF5 data: " + name);
        std::vector<hsize_t> dims(shape.begin(), shape.end());
        std::vector<hsize_t> offsets(start.begin(), start.end());
        std::vector<hsize_t> counts(count.begin(), count.end());
        Handle space(shape.empty() ? H5Screate(H5S_SCALAR) :
            H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
        Handle creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
        if (shape.size() >= 2 && bytes >= options_.minimum_bytes) {
            auto chunks = dims;
            for (size_t d = 0; d + 2 < chunks.size(); ++d) chunks[d] = 1;
            chunks[chunks.size()-2] = std::min<hsize_t>(chunks[chunks.size()-2], 128);
            chunks.back() = std::min<hsize_t>(chunks.back(), 512);
            check(H5Pset_chunk(creation, static_cast<int>(chunks.size()), chunks.data()), "chunk");
            check(H5Pset_shuffle(creation), "shuffle");
            check(H5Pset_deflate(creation, options_.level), "gzip");
            check(H5Pset_fletcher32(creation), "checksum");
        }
        Handle links(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
        check(H5Pset_create_intermediate_group(links, 1), "nested groups");
        Handle dataset(H5Dcreate2(group_, name.c_str(), type<T>(), space, links,
            creation, H5P_DEFAULT), H5Dclose);
        if (name.find('/') != std::string::npos)
            string_attribute(dataset, "ADIOSName", name);
        Handle memory((shape.empty() || empty) ? H5Screate(H5S_SCALAR) :
            H5Screate_simple(static_cast<int>(counts.size()), counts.data(), nullptr), H5Sclose);
        if (empty) {
            check(H5Sselect_none(space), "empty file selection");
            check(H5Sselect_none(memory), "empty memory selection");
        } else if (!shape.empty()) {
            check(H5Sselect_hyperslab(space, H5S_SELECT_SET, offsets.data(), nullptr,
                counts.data(), nullptr), "hyperslab");
        }
        Handle transfer(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
        check(H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE), "collective transfer");
        const T dummy{};
        check(H5Dwrite(dataset, type<T>(), memory, space, transfer, empty ? &dummy : data),
            "write dataset");
    }

    static void string_attribute(hid_t object, const std::string& name, const std::string& value) {
        Handle space(H5Screate(H5S_SCALAR), H5Sclose);
        Handle datatype(H5Tcopy(H5T_C_S1), H5Tclose);
        check(H5Tset_size(datatype, value.size()+1), "string size");
        check(H5Tset_strpad(datatype, H5T_STR_NULLTERM), "string padding");
        Handle attr(H5Acreate2(object, name.c_str(), datatype, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
        check(H5Awrite(attr, datatype, value.c_str()), "string attribute");
    }
    void attribute(const std::string& name, const std::string& value) {
        string_attribute(file_, name, value);
    }
    template<class T> void attribute(const std::string& name, const std::vector<T>& values) {
        if (values.empty()) throw std::invalid_argument("Empty HDF5 attribute: " + name);
        const hsize_t n = values.size();
        Handle space(n == 1 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &n, nullptr), H5Sclose);
        Handle attr(H5Acreate2(file_, name.c_str(), type<T>(), space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
        check(H5Awrite(attr, type<T>(), values.data()), "numeric attribute");
    }
    void close() {
        check(H5Fflush(file_, H5F_SCOPE_GLOBAL), "flush");
        group_.close();
        file_.close();
    }
};
} // namespace VVM::IO
