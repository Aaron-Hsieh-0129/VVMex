#ifndef VVM_IO_BP5_PATH_POLICY_HPP
#define VVM_IO_BP5_PATH_POLICY_HPP

#include <filesystem>
#include <string>

namespace VVM::IO::BP5 {

std::filesystem::path prepare_bp5_dataset_path(
    const std::string& output_dir,
    const std::string& prefix,
    bool overwrite);

} // namespace VVM::IO::BP5

#endif
