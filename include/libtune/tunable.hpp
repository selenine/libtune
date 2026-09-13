#pragma once

#include <algorithm>
#include <cereal/archives/binary.hpp>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace tune {
class Tunable {
    public:
    Tunable(std::span<float> tunables, float smoothing = 1.0f) : smoothing_(smoothing) {
        data_.assign_range(tunables);
        grad_.resize(tunables.size());
    }

    Tunable(std::span<int> tunables, float smoothing = 1.0f) {
        data_.assign_range(std::views::transform(tunables, [](int x) {
            return static_cast<float>(x);
        }));
        grad_.resize(tunables.size());
    }

    auto numel() -> int { return data_.size(); }
    auto zero_grad() -> void { std::ranges::fill(grad_, static_cast<float>(0.0)); }
    auto data() -> std::vector<float>& { return data_; }
    auto grad() -> std::vector<float>& { return grad_; }
    auto update(std::span<float>& vals) -> void { data_.assign_range(vals); }

    auto serialize(const std::string& str) -> void {
        auto path = std::filesystem::path(str);
        path.replace_extension(".bin");
        auto out = std::ofstream(path);
        auto archive = cereal::BinaryOutputArchive(out);
        archive(data_);
    }

    auto deserialize(const std::string& str) -> void {
        auto path = std::filesystem::path(str);
        path.replace_extension(".bin");
        auto in = std::ifstream(path);
        auto archive = cereal::BinaryInputArchive(in);
        archive(data_);
    }

    private:
    std::vector<float> data_;
    std::vector<float> grad_;
    float smoothing_;
};
}  // namespace tune
