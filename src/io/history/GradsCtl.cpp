#include "io/history/GradsCtl.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace VVM::IO {
namespace {

constexpr VVM::Real earth_radius_m = VVM::real(6.37e6);
constexpr VVM::Real pi = VVM::real(3.141592653589793238462643383279502884);
constexpr std::size_t grads_name_limit = 15;
constexpr std::size_t levels_per_line = 15;

std::string format_axis_number(VVM::Real value) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(7) << value;
    std::string formatted = ss.str();
    if (value > VVM::real(0.0) && value < VVM::real(1.0) && formatted.rfind("0.", 0) == 0) {
        formatted.erase(0, 1);
    }
    return formatted;
}

GradsAxis centered_lonlat_axis(int points, VVM::Real spacing) {
    GradsAxis axis;
    axis.increment = spacing / earth_radius_m / (real(2.0) * pi) * real(360.0);
    axis.start = (real(0.5) - real(0.5) * static_cast<VVM::Real>(points)) * axis.increment;
    return axis;
}

} // namespace

std::pair<GradsAxis, GradsAxis> grads_horizontal_axes(
    const Core::Grid& grid,
    const Core::State& state,
    bool use_taiwanvvm_coordinates,
    MPI_Comm comm) {
    GradsAxis x_axis = centered_lonlat_axis(grid.get_global_points_x(), grid.get_dx());
    GradsAxis y_axis = centered_lonlat_axis(grid.get_global_points_y(), grid.get_dy());

    if (!use_taiwanvvm_coordinates) {
        return {x_axis, y_axis};
    }

    const int h = grid.get_halo_cells();
    const int local_start_x = grid.get_local_physical_start_x();
    const int local_end_x = grid.get_local_physical_end_x();
    const int local_start_y = grid.get_local_physical_start_y();
    const int local_end_y = grid.get_local_physical_end_y();

    std::array<VVM::Real, 4> local_values = {
        real(0.0), real(0.0), real(0.0), real(0.0)
    };
    std::array<int, 4> local_flags = {0, 0, 0, 0};

    auto owns_point = [&](int global_y, int global_x) {
        return local_start_y <= global_y && global_y <= local_end_y &&
               local_start_x <= global_x && global_x <= local_end_x;
    };

    if (owns_point(0, 0)) {
        const auto lon_host = state.get_field<2>("lon").get_host_data();
        const auto lat_host = state.get_field<2>("lat").get_host_data();
        const int j = h - local_start_y;
        const int i = h - local_start_x;
        local_values[0] = lon_host(j, i);
        local_values[2] = lat_host(j, i);
        local_flags[0] = 1;
        local_flags[2] = 1;
    }

    if (grid.get_global_points_x() > 1 && owns_point(0, 1)) {
        const auto lon_host = state.get_field<2>("lon").get_host_data();
        const int j = h - local_start_y;
        const int i = h + 1 - local_start_x;
        local_values[1] = lon_host(j, i);
        local_flags[1] = 1;
    }

    if (grid.get_global_points_y() > 1 && owns_point(1, 0)) {
        const auto lat_host = state.get_field<2>("lat").get_host_data();
        const int j = h + 1 - local_start_y;
        const int i = h - local_start_x;
        local_values[3] = lat_host(j, i);
        local_flags[3] = 1;
    }

    std::array<VVM::Real, 4> global_values;
    std::array<int, 4> global_flags;
    MPI_Allreduce(local_values.data(), global_values.data(), 4, VVM_MPI_REAL, MPI_SUM, comm);
    MPI_Allreduce(local_flags.data(), global_flags.data(), 4, MPI_INT, MPI_SUM, comm);

    if (global_flags[0] > 0) x_axis.start = global_values[0] / static_cast<VVM::Real>(global_flags[0]);
    if (global_flags[2] > 0) y_axis.start = global_values[2] / static_cast<VVM::Real>(global_flags[2]);
    if (global_flags[1] > 0) {
        x_axis.increment = global_values[1] / static_cast<VVM::Real>(global_flags[1]) - x_axis.start;
    }
    if (global_flags[3] > 0) {
        y_axis.increment = global_values[3] / static_cast<VVM::Real>(global_flags[3]) - y_axis.start;
    }

    return {x_axis, y_axis};
}

std::string grads_start_time(int start_hour) {
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(2) << start_hour << "z01JAN1998";
    return ss.str();
}

std::string grads_time_increment(VVM::Real output_interval_s) {
    const auto minutes = std::max<long long>(
        1,
        static_cast<long long>(std::llround(output_interval_s / real(60.0)))
    );

    if (minutes % 60 == 0) return std::to_string(minutes / 60) + "hr";
    return std::to_string(minutes) + "mn";
}

std::string unique_grads_variable_name(
    const std::string& field_name,
    std::unordered_set<std::string>& taken) {
    std::string name;
    name.reserve(field_name.size());
    for (const unsigned char c : field_name) {
        if (std::isalnum(c) || c == '_') {
            name.push_back(static_cast<char>(std::tolower(c)));
        } else {
            name.push_back('_');
        }
    }
    if (name.empty() || std::isdigit(static_cast<unsigned char>(name.front())) ||
        name.front() == '_') {
        name.insert(name.begin(), 'v');
    }
    if (name.size() > grads_name_limit) name.resize(grads_name_limit);

    std::string candidate = name;
    for (int suffix = 2; !taken.insert(candidate).second; ++suffix) {
        const std::string tail = std::to_string(suffix);
        candidate = name.substr(0, std::min(name.size(), grads_name_limit - tail.size())) + tail;
    }
    return candidate;
}

void write_grads_ctl(const std::filesystem::path& path, const GradsCtl& ctl) {
    std::ofstream file(path);
    if (!file.is_open()) return;

    file << "DSET " << ctl.dset << "\n";
    file << "DTYPE " << ctl.dtype << "\n";
    if (ctl.templated) file << "OPTIONS template\n";
    file << "TITLE " << ctl.title << "\n";
    file << "UNDEF " << ctl.undef << "\n";
    file << "XDEF " << ctl.nx << " LINEAR "
         << format_axis_number(ctl.x.start) << " "
         << format_axis_number(ctl.x.increment) << "\n";
    file << "YDEF " << ctl.ny << " LINEAR "
         << format_axis_number(ctl.y.start) << " "
         << format_axis_number(ctl.y.increment) << "\n";
    file << "ZDEF " << ctl.z_levels.size() << " LEVELS ";
    for (std::size_t k = 0; k < ctl.z_levels.size(); ++k) {
        file << static_cast<int>(ctl.z_levels[k]);
        if (k + 1 == ctl.z_levels.size()) continue;
        file << ((k + 1) % levels_per_line == 0 ? "\n" : " ");
    }
    file << "\n";
    file << "TDEF " << ctl.time_count << " LINEAR " << ctl.time_start << " "
         << ctl.time_increment << "\n";
    file << "\n";
    for (const auto& note : ctl.notes) file << "* " << note << "\n";

    file << "VARS " << ctl.variables.size() << "\n";
    for (const auto& variable : ctl.variables) {
        file << variable.dataset_name;
        if (!variable.grads_name.empty() && variable.grads_name != variable.dataset_name) {
            file << "=>" << variable.grads_name;
        }
        file << " " << variable.levels << " " << variable.dimensions << " "
             << variable.description << "\n";
    }
    file << "ENDVARS\n";
}

} // namespace VVM::IO
