#ifndef VVM_IO_PNETCDF_RESTART_METADATA_HPP
#define VVM_IO_PNETCDF_RESTART_METADATA_HPP

#include "utils/RestartTime.hpp"

#include <mpi.h>
#include <pnetcdf.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace VVM {
namespace IO {

// Reading the restart clock out of a NetCDF restart file.
inline bool is_elapsed_seconds_units(std::string units) {
    units.erase(units.begin(),
                std::find_if(units.begin(), units.end(),
                             [](unsigned char c) { return !std::isspace(c); }));
    while (!units.empty() &&
           (std::isspace(static_cast<unsigned char>(units.back())) || units.back() == '\0')) {
        units.pop_back();
    }
    std::transform(units.begin(), units.end(), units.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return units == "s" || units == "sec" || units == "secs" ||
           units == "second" || units == "seconds";
}

inline bool read_pnetcdf_text_attribute(int ncid, int varid, const std::string& name,
                                        std::string& out) {
    MPI_Offset length = 0;
    if (ncmpi_inq_attlen(ncid, varid, name.c_str(), &length) != NC_NOERR) return false;
    if (length < 0) return false;

    std::vector<char> buffer(static_cast<std::size_t>(length) + 1, '\0');
    if (ncmpi_get_att_text(ncid, varid, name.c_str(), buffer.data()) != NC_NOERR) return false;

    out.assign(buffer.data());
    return true;
}

// A scalar variable, or a length-1 first dimension. Anything else is not a
// restart scalar and is reported as absent.
inline bool read_pnetcdf_scalar_double(int ncid, const std::string& name, double& out) {
    int varid = -1;
    if (ncmpi_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR) return false;

    int ndims = 0;
    if (ncmpi_inq_varndims(ncid, varid, &ndims) != NC_NOERR) return false;

    if (ndims == 0) {
        return ncmpi_get_var_double_all(ncid, varid, &out) == NC_NOERR;
    }
    if (ndims == 1) {
        MPI_Offset start[1] = {0};
        MPI_Offset count[1] = {1};
        return ncmpi_get_vara_double_all(ncid, varid, start, count, &out) == NC_NOERR;
    }
    return false;
}

inline bool read_pnetcdf_scalar_int64(int ncid, const std::string& name, long long& out) {
    int varid = -1;
    if (ncmpi_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR) return false;

    int ndims = 0;
    if (ncmpi_inq_varndims(ncid, varid, &ndims) != NC_NOERR) return false;

    if (ndims == 0) {
        return ncmpi_get_var_longlong_all(ncid, varid, &out) == NC_NOERR;
    }
    if (ndims == 1) {
        MPI_Offset start[1] = {0};
        MPI_Offset count[1] = {1};
        return ncmpi_get_vara_longlong_all(ncid, varid, start, count, &out) == NC_NOERR;
    }
    return false;
}

// Collect whatever restart metadata an already-open NetCDF file carries.
inline VVM::Utils::RestartFileMetadata read_pnetcdf_restart_metadata(int ncid) {
    VVM::Utils::RestartFileMetadata metadata;

    double time_s = 0.0;
    if (read_pnetcdf_scalar_double(ncid, "model_time_s", time_s)) {
        metadata.has_time = true;
        metadata.time_s = time_s;
    } else if (ncmpi_get_att_double(ncid, NC_GLOBAL, "model_time_s", &time_s) == NC_NOERR) {
        metadata.has_time = true;
        metadata.time_s = time_s;
    } else {
        int varid = -1;
        std::string units;
        if (ncmpi_inq_varid(ncid, "time", &varid) == NC_NOERR &&
            read_pnetcdf_text_attribute(ncid, varid, "units", units) &&
            is_elapsed_seconds_units(units) &&
            read_pnetcdf_scalar_double(ncid, "time", time_s)) {
            metadata.has_time = true;
            metadata.time_s = time_s;
        }
    }

    long long step = 0;
    if (read_pnetcdf_scalar_int64(ncid, "model_step", step)) {
        metadata.has_step = true;
        metadata.step = step;
    } else if (ncmpi_get_att_longlong(ncid, NC_GLOBAL, "model_step", &step) == NC_NOERR) {
        metadata.has_step = true;
        metadata.step = step;
    }

    return metadata;
}

} // namespace IO
} // namespace VVM

#endif // VVM_IO_PNETCDF_RESTART_METADATA_HPP
