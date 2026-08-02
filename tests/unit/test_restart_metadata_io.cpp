// Unit tests for getting the restart clock out of a real file
// (src/io/Hdf5RestartMetadata.hpp, src/io/PnetcdfRestartMetadata.hpp).
//
// tests/unit/test_restart_time.cpp pins the decision logic; this one pins the
// part that touches bytes on disk. The headline case is the regression that
// motivated all of it:
//
//     write a restart file -> rename it -> load the renamed file
//
// and the recovered time and step must not move. The file name used to be the
// only thing the model looked at, so this went wrong silently.
//
// Both container formats are covered, because fixing only HDF5 would leave the
// ".nc" restart path deriving its clock from the file name:
//
//   HDF5    /Step0/model_time_s, /Step0/model_step, legacy /Step0/time
//           (also at the root, for files written without the ADIOS2 step group)
//   NetCDF  model_time_s / model_step as scalar variables or as global
//           attributes; a bare "time" only when its units say plain seconds,
//           since a NetCDF "time" is just as often a calendar coordinate.
//
// Needs MPI because PnetCDF does; run it at more than one rank and it also
// checks that every rank recovers bit-identical values.
//
// Reporting goes through <cstdio> rather than <iostream> on purpose -- see the
// note in test_sst_path.cpp about the NVHPC/libstdc++ mismatch.

#include "io/Hdf5RestartMetadata.hpp"
#include "io/PnetcdfRestartMetadata.hpp"
#include "utils/RestartTime.hpp"

#include <hdf5.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

namespace fs = std::filesystem;

using VVM::IO::read_hdf5_restart_metadata;
using VVM::IO::read_pnetcdf_restart_metadata;
using VVM::Utils::resolve_restart_time;
using VVM::Utils::RestartFileMetadata;
using VVM::Utils::RestartTimePolicy;
using VVM::Utils::RestartTimeSource;

namespace {

int rank = 0;
int nranks = 1;
int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "[rank %d] FAIL: %s\n", rank, what.c_str());
        ++failures;
    }
}

void check_metadata(const RestartFileMetadata& m, bool has_time, double time_s,
                    bool has_step, long long step, const std::string& what) {
    check(m.has_time == has_time,
          what + ": has_time is " + (m.has_time ? "true" : "false"));
    check(m.has_step == has_step,
          what + ": has_step is " + (m.has_step ? "true" : "false"));
    if (has_time && m.has_time) {
        check(m.time_s == time_s, what + ": time " + std::to_string(m.time_s) +
                                      " != " + std::to_string(time_s));
    }
    if (has_step && m.has_step) {
        check(m.step == step, what + ": step " + std::to_string(m.step) +
                                  " != " + std::to_string(step));
    }
}

// Every rank must recover the same numbers from the same file, bit for bit --
// a restart where one rank disagrees about the clock is worse than one that
// stops.
void check_identical_across_ranks(const RestartFileMetadata& m, const std::string& what) {
    if (nranks < 2) return;

    double time_min = m.time_s, time_max = m.time_s;
    long long step_min = m.step, step_max = m.step;
    int flags = (m.has_time ? 1 : 0) | (m.has_step ? 2 : 0);
    int flags_min = flags, flags_max = flags;

    MPI_Allreduce(MPI_IN_PLACE, &time_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &time_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &step_min, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &step_max, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &flags_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &flags_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    check(time_min == time_max, what + ": ranks disagree about the restart time");
    check(step_min == step_max, what + ": ranks disagree about the restart step");
    check(flags_min == flags_max, what + ": ranks disagree about what the file contains");
}

// --- HDF5 writing helpers --------------------------------------------------

void write_hdf5_scalar_double(hid_t where, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate2(where, name, H5T_IEEE_F64LE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
}

void write_hdf5_scalar_float(hid_t where, const char* name, float value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate2(where, name, H5T_IEEE_F32LE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
}

void write_hdf5_scalar_int64(hid_t where, const char* name, int64_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate2(where, name, H5T_STD_I64LE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
    H5Sclose(space);
}

// A 3-D field, so the file is not just scalars: the metadata read must ignore
// everything that is not a one-element dataset.
void write_hdf5_dummy_field(hid_t where) {
    const hsize_t dims[3] = {2, 2, 2};
    double values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    hid_t space = H5Screate_simple(3, dims, nullptr);
    hid_t dataset = H5Dcreate2(where, "th", H5T_IEEE_F64LE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);
    H5Dclose(dataset);
    H5Sclose(space);
}

// The layout ADIOS2's HDF5 engine produces: everything under /Step0.
void write_hdf5_output_file(const fs::path& path, bool with_model_time, bool with_model_step,
                            bool with_legacy_time, double time_s, int64_t step) {
    hid_t file = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t group = H5Gcreate2(file, "/Step0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    write_hdf5_dummy_field(group);
    if (with_legacy_time) write_hdf5_scalar_double(group, "time", time_s);
    if (with_model_time) write_hdf5_scalar_double(group, "model_time_s", time_s);
    if (with_model_step) write_hdf5_scalar_int64(group, "model_step", step);

    H5Gclose(group);
    H5Fclose(file);
}

// --- NetCDF writing helpers ------------------------------------------------

void write_netcdf_file(MPI_Comm comm, const fs::path& path, bool with_scalar_vars,
                       bool with_global_attrs, const char* time_units, double time_s,
                       long long step) {
    int ncid = -1;
    // CDF-5 (NC_64BIT_DATA): NC_INT64 does not exist in the classic formats.
    if (ncmpi_create(comm, path.string().c_str(), NC_CLOBBER | NC_64BIT_DATA,
                     MPI_INFO_NULL, &ncid) != NC_NOERR) {
        check(false, "could not create NetCDF file " + path.string());
        return;
    }

    int nz_dim = -1;
    ncmpi_def_dim(ncid, "nz", 4, &nz_dim);

    int field_var = -1;
    ncmpi_def_var(ncid, "th", NC_DOUBLE, 1, &nz_dim, &field_var);

    int time_var = -1;
    int step_var = -1;
    int units_var = -1;
    if (with_scalar_vars) {
        ncmpi_def_var(ncid, "model_time_s", NC_DOUBLE, 0, nullptr, &time_var);
        ncmpi_def_var(ncid, "model_step", NC_INT64, 0, nullptr, &step_var);
    }
    if (time_units != nullptr) {
        ncmpi_def_var(ncid, "time", NC_DOUBLE, 0, nullptr, &units_var);
        ncmpi_put_att_text(ncid, units_var, "units", std::string(time_units).size(), time_units);
    }
    if (with_global_attrs) {
        ncmpi_put_att_double(ncid, NC_GLOBAL, "model_time_s", NC_DOUBLE, 1, &time_s);
        ncmpi_put_att_longlong(ncid, NC_GLOBAL, "model_step", NC_INT64, 1, &step);
    }

    ncmpi_enddef(ncid);

    const double field[4] = {0.0, 1.0, 2.0, 3.0};
    MPI_Offset start[1] = {0};
    MPI_Offset count[1] = {4};
    ncmpi_put_vara_double_all(ncid, field_var, start, count, field);

    if (with_scalar_vars) {
        ncmpi_put_var_double_all(ncid, time_var, &time_s);
        ncmpi_put_var_longlong_all(ncid, step_var, &step);
    }
    if (time_units != nullptr) {
        ncmpi_put_var_double_all(ncid, units_var, &time_s);
    }

    ncmpi_close(ncid);
}

RestartFileMetadata read_netcdf_metadata(MPI_Comm comm, const fs::path& path) {
    int ncid = -1;
    if (ncmpi_open(comm, path.string().c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid) != NC_NOERR) {
        check(false, "could not open NetCDF file " + path.string());
        return RestartFileMetadata{};
    }
    const RestartFileMetadata metadata = read_pnetcdf_restart_metadata(ncid);
    ncmpi_close(ncid);
    return metadata;
}

RestartFileMetadata read_hdf5_metadata(const fs::path& path) {
    RestartFileMetadata metadata;
    if (!VVM::IO::read_hdf5_restart_metadata(path.string(), metadata)) {
        check(false, "could not open HDF5 file " + path.string());
    }
    return metadata;
}

RestartTimePolicy dt_policy(double dt_s) {
    RestartTimePolicy policy;
    policy.dt_s = dt_s;
    return policy;
}

// --- the tests -------------------------------------------------------------

void test_hdf5(const fs::path& dir) {
    const double time_s = 7200.0;
    const int64_t step = 3600;

    // 1. both time and step.
    const fs::path both = dir / "vvm_output_000002.h5";
    if (rank == 0) write_hdf5_output_file(both, true, true, true, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata both_meta = read_hdf5_metadata(both);
    check_metadata(both_meta, true, time_s, true, step, "HDF5 file with time and step");
    check_identical_across_ranks(both_meta, "HDF5 file with time and step");

    // 2. time, no step -- what files written before model_step look like.
    const fs::path legacy = dir / "vvm_output_000002_legacy.h5";
    if (rank == 0) write_hdf5_output_file(legacy, false, false, true, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata legacy_meta = read_hdf5_metadata(legacy);
    check_metadata(legacy_meta, true, time_s, false, 0, "HDF5 file with legacy time only");
    check_identical_across_ranks(legacy_meta, "HDF5 file with legacy time only");

    const auto legacy_resolved = resolve_restart_time(legacy_meta, dt_policy(2.0), legacy.string());
    check(legacy_resolved.ok && legacy_resolved.time_s == time_s && legacy_resolved.step == step,
          "a legacy time-only HDF5 file still resolves to the right step");

    // 3 & 4 & 5. THE regression: write -> rename -> load. The new name's digits
    // say 999 (which under the old rule meant 999 * file_interval_s), and the
    // recovered clock must not notice.
    const fs::path renamed = dir / "vvm_output_000999.h5";
    if (rank == 0) {
        std::error_code ec;
        fs::rename(both, renamed, ec);
        check(!ec, "could not rename the restart file: " + ec.message());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata renamed_meta = read_hdf5_metadata(renamed);
    check_metadata(renamed_meta, true, time_s, true, step,
                   "renaming an HDF5 restart file does not change its metadata");
    check_identical_across_ranks(renamed_meta, "renamed HDF5 restart file");

    RestartTimePolicy fallback_policy = dt_policy(2.0);
    fallback_policy.allow_filename_fallback = true;   // even with the old rule available
    fallback_policy.filename_interval_s = 600.0;      // 999 * 600 = 599400 s, if it were used
    const auto renamed_resolved =
        resolve_restart_time(renamed_meta, fallback_policy, renamed.string());
    check(renamed_resolved.ok, "the renamed file resolves");
    check(renamed_resolved.time_s == time_s && renamed_resolved.step == step,
          "the renamed file resolves to the same clock as before the rename");
    check(renamed_resolved.source == RestartTimeSource::FileTimeAndStep,
          "the file's own metadata is what was used, not the name");

    // 6. a file with no clock at all.
    const fs::path bare = dir / "vvm_output_000002_bare.h5";
    if (rank == 0) write_hdf5_output_file(bare, false, false, false, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata bare_meta = read_hdf5_metadata(bare);
    check_metadata(bare_meta, false, 0.0, false, 0, "HDF5 file with no clock");
    check(!resolve_restart_time(bare_meta, dt_policy(2.0), bare.string()).ok,
          "a clockless HDF5 file is refused rather than read from its name");

    // 7. scalars at the root instead of under /Step0.
    const fs::path root_level = dir / "root_level.h5";
    if (rank == 0) {
        hid_t file = H5Fcreate(root_level.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        write_hdf5_scalar_double(file, "model_time_s", time_s);
        write_hdf5_scalar_int64(file, "model_step", step);
        H5Fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    check_metadata(read_hdf5_metadata(root_level), true, time_s, true, step,
                   "HDF5 scalars at the file root");

    // 8. a single-precision time still reads back as the same number.
    const fs::path float_time = dir / "float_time.h5";
    if (rank == 0) {
        hid_t file = H5Fcreate(float_time.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t group = H5Gcreate2(file, "/Step0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        write_hdf5_scalar_float(group, "time", static_cast<float>(time_s));
        H5Gclose(group);
        H5Fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    check_metadata(read_hdf5_metadata(float_time), true, time_s, false, 0,
                   "a single-precision HDF5 time converts on read");
}

void test_netcdf(const fs::path& dir) {
    const double time_s = 7200.0;
    const long long step = 3600;

    // 1. scalar model_time_s / model_step variables.
    const fs::path vars = dir / "restart_000002.nc";
    write_netcdf_file(MPI_COMM_WORLD, vars, true, false, nullptr, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata var_meta = read_netcdf_metadata(MPI_COMM_WORLD, vars);
    check_metadata(var_meta, true, time_s, true, step, "NetCDF scalar restart variables");
    check_identical_across_ranks(var_meta, "NetCDF scalar restart variables");

    // 2. the same information as global attributes.
    const fs::path attrs = dir / "restart_attrs_000002.nc";
    write_netcdf_file(MPI_COMM_WORLD, attrs, false, true, nullptr, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);
    check_metadata(read_netcdf_metadata(MPI_COMM_WORLD, attrs), true, time_s, true, step,
                   "NetCDF global restart attributes");

    // 3. rename: same file, different digits, same clock.
    const fs::path renamed = dir / "restart_000999.nc";
    if (rank == 0) {
        std::error_code ec;
        fs::rename(vars, renamed, ec);
        check(!ec, "could not rename the NetCDF restart file: " + ec.message());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata renamed_meta = read_netcdf_metadata(MPI_COMM_WORLD, renamed);
    check_metadata(renamed_meta, true, time_s, true, step,
                   "renaming a NetCDF restart file does not change its metadata");
    check_identical_across_ranks(renamed_meta, "renamed NetCDF restart file");

    // 4. a "time" variable in plain seconds is accepted...
    const fs::path seconds = dir / "seconds_time.nc";
    write_netcdf_file(MPI_COMM_WORLD, seconds, false, false, "seconds", time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);
    check_metadata(read_netcdf_metadata(MPI_COMM_WORLD, seconds), true, time_s, false, 0,
                   "a NetCDF 'time' in seconds is an elapsed time");

    // ... and a calendar "time" is not. Reading "hours since ..." as elapsed
    // seconds is precisely the silent wrong answer this must not give.
    const fs::path calendar = dir / "calendar_time.nc";
    write_netcdf_file(MPI_COMM_WORLD, calendar, false, false,
                      "hours since 2025-10-07 00:00:00", time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);

    const RestartFileMetadata calendar_meta = read_netcdf_metadata(MPI_COMM_WORLD, calendar);
    check_metadata(calendar_meta, false, 0.0, false, 0,
                   "a NetCDF calendar 'time' is not an elapsed time");

    // Such a file needs the explicit legacy policy, and says so.
    check(!resolve_restart_time(calendar_meta, dt_policy(2.0), calendar.string()).ok,
          "a NetCDF file with only a calendar time is refused");

    RestartTimePolicy legacy_policy = dt_policy(2.0);
    legacy_policy.has_legacy_time = true;
    legacy_policy.legacy_time_s = time_s;
    const auto legacy_resolved =
        resolve_restart_time(calendar_meta, legacy_policy, calendar.string());
    check(legacy_resolved.ok && legacy_resolved.time_s == time_s &&
              legacy_resolved.step == step &&
              legacy_resolved.source == RestartTimeSource::LegacyConfig,
          "restart.legacy_time_s carries a legacy NetCDF restart file");
    check(!legacy_resolved.warnings.empty(), "and it warns while doing so");

    // 5. no clock at all.
    const fs::path bare = dir / "bare_000002.nc";
    write_netcdf_file(MPI_COMM_WORLD, bare, false, false, nullptr, time_s, step);
    MPI_Barrier(MPI_COMM_WORLD);
    check_metadata(read_netcdf_metadata(MPI_COMM_WORLD, bare), false, 0.0, false, 0,
                   "NetCDF file with no clock");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One directory, agreed on by every rank.
    long long tag = 0;
    if (rank == 0) tag = static_cast<long long>(getpid());
    MPI_Bcast(&tag, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    const fs::path dir =
        fs::temp_directory_path() / ("vvm_restart_metadata_test_" + std::to_string(tag));
    if (rank == 0) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            std::fprintf(stderr, "FAIL: cannot create %s: %s\n", dir.string().c_str(),
                         ec.message().c_str());
            ++failures;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (failures == 0) {
        test_hdf5(dir);
        test_netcdf(dir);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    int total_failures = 0;
    MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total_failures != 0) {
            std::fprintf(stderr, "test_restart_metadata_io: %d check(s) failed over %d rank(s)\n",
                         total_failures, nranks);
        } else {
            std::printf("test_restart_metadata_io: all checks passed on %d rank(s)\n", nranks);
        }
    }

    MPI_Finalize();
    return total_failures == 0 ? 0 : 1;
}
