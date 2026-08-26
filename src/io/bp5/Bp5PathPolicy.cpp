#include "Bp5PathPolicy.hpp"

#include <stdexcept>
#include <system_error>

namespace VVM::IO::BP5 {

std::filesystem::path prepare_bp5_dataset_path(
    const std::string& output_dir,
    const std::string& prefix,
    ExistingDatasetPolicy policy) {
    if (output_dir.empty()) {
        throw std::invalid_argument("output.output_dir must not be empty for BP5.");
    }
    const std::filesystem::path prefix_path(prefix);
    if (prefix.empty() || prefix == "." || prefix == ".." ||
        prefix_path.has_parent_path() || prefix_path.filename() != prefix_path) {
        throw std::invalid_argument(
            "output.output_filename_prefix must be a simple non-empty filename for BP5.");
    }

    const std::string dataset_name =
        prefix_path.extension() == ".bp" ? prefix : prefix + ".bp";
    const std::filesystem::path dataset_path =
        std::filesystem::path(output_dir) / dataset_name;
    if (dataset_path.extension() != ".bp" || dataset_path.filename() == ".bp") {
        throw std::invalid_argument("refusing unsafe BP5 dataset target");
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        throw std::runtime_error("cannot create output directory: " + ec.message());
    }
    const auto status = std::filesystem::symlink_status(dataset_path, ec);
    if (ec == std::errc::no_such_file_or_directory) ec.clear();
    if (ec) {
        throw std::runtime_error("cannot inspect dataset path: " + ec.message());
    }
    const bool exists = status.type() != std::filesystem::file_type::not_found;
    if (exists && policy == ExistingDatasetPolicy::Error) {
        throw std::runtime_error(
            "dataset already exists and output.bp5.existing_dataset is 'error'");
    }
    if (exists) {
        if (std::filesystem::is_symlink(status)) {
            throw std::runtime_error("refusing to use a symlink dataset path");
        }
        if (policy == ExistingDatasetPolicy::Append) {
            if (!std::filesystem::is_directory(status)) {
                throw std::runtime_error("BP5 append target is not a dataset directory");
            }
            return dataset_path;
        }
        if (policy == ExistingDatasetPolicy::Replace) {
            std::filesystem::remove_all(dataset_path, ec);
            if (ec) {
                throw std::runtime_error("cannot remove existing BP5 dataset: " + ec.message());
            }
        }
    } else if (policy == ExistingDatasetPolicy::Append) {
        throw std::runtime_error("BP5 append target does not exist");
    }
    return dataset_path;
}

} // namespace VVM::IO::BP5
