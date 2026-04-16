#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>

namespace phash_cuda {

std::vector<std::array<uint8_t, 8>> compute_phash(
    const std::vector<std::string>& image_paths,
    int resize = 32,
    int low_freq = 8,
    int batch_size = 128
);

}
