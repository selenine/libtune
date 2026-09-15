#pragma once

#include <cmath>
#include <eve/module/core.hpp>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../detail/utils.hpp"
#include "../tunable.hpp"

namespace tune {
namespace detail {
struct AdamState {
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    AdamState(T&... tunables) {
        ((momentum_.emplace_back(std::vector<float>(tunables.numel())),
          velocity_.emplace_back(std::vector<float>(tunables.numel()))),
         ...);
    }

    std::vector<std::vector<float>> momentum_{};
    std::vector<std::vector<float>> velocity_{};
    int step = 0;
};
}  // namespace detail

namespace optim {
struct AdamOptions {
    float lr = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.4f;
    float eps = 1e-7f;
    float lambda = 0.0f;

    bool minimize = false;
    bool cautious = false;
    bool adabelief = false;
};

class Adam {
    public:
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    Adam(const AdamOptions& options, T&... tunables) : options_(options), state_(tunables...) {
        (tunables_.emplace_back(tunables), ...);
    }

    constexpr auto zero_grad() noexcept -> void {
        for (auto& tnbl : tunables_) {
            auto t = tnbl.get();
            t.zero_grad();
        }
    }

    // TODO: also noexcept this
    constexpr auto step() -> void {
        for (auto&& [index, tnbl] : std::views::enumerate(tunables_)) {
            auto& t = tnbl.get();
            auto& moms = state_.momentum_[index];
            auto& vels = state_.velocity_[index];
            auto& grads = t.grad();
            auto& datas = t.data();

            constexpr auto VEC_SIZE = eve::wide<float>::size();
            constexpr auto CHUNK_SIZE = VEC_SIZE * detail::UNROLL;
            const auto N_CHUNKS = t.numel() / VEC_SIZE * detail::UNROLL;
            const auto BLOCKED = CHUNK_SIZE * N_CHUNKS;
            const auto REM = t.numel() - BLOCKED;

            for (const auto& i : std::views::iota(0, N_CHUNKS)) {
                detail::unroll<detail::UNROLL>([&]<size_t idx>() {
                    constexpr auto OFFSET = idx * VEC_SIZE;

                    auto data = eve::wide<float>(&datas[i * CHUNK_SIZE + OFFSET]);
                    auto grad = [&]() noexcept -> eve::wide<float> {
                        auto grad = eve::wide<float>(&grads[i * CHUNK_SIZE + OFFSET]);
                        if (options_.minimize) {
                            grad = -grad;
                        }

                        auto beta1 = eve::wide<float>(options_.beta1);
                        auto beta2 = eve::wide<float>(options_.beta2);

                        auto mom = eve::wide<float>(&moms[i * CHUNK_SIZE + OFFSET]);
                        mom = eve::fma(beta1, mom, grad);
                        mom = eve::fnma(beta1, grad, mom);
                        mom /= eve::wide<float>(std::pow((1 - options_.beta1), state_.step));

                        auto vel = eve::wide<float>(&vels[i * CHUNK_SIZE + OFFSET]);
                        auto sqr = (options_.adabelief) ? (grad - mom) * (grad - mom) +
                                                              eve::wide<float>(options_.eps)
                                                        : grad * grad;

                        vel = eve::fma(beta2, vel, sqr);
                        vel = eve::fnma(beta2, sqr, vel);
                        vel /= eve::wide<float>(std::pow((1 - options_.beta2), state_.step));

                        auto upd = mom / (eve::sqrt(vel) + eve::wide<float>(options_.eps));
                        if (options_.cautious) {
                            auto grad_sign = (grad > 0);
                            auto upd_sign = (upd > 0);
                            auto mask = (grad_sign == upd_sign);
                            upd = eve::if_else(mask, upd, eve::zero);
                        }

                        eve::store(mom, &moms[i * CHUNK_SIZE + OFFSET]);
                        eve::store(vel, &vels[i * CHUNK_SIZE + OFFSET]);

                        return grad;
                    }();

                    data = eve::fma(eve::wide<float>(options_.lr), grad, data);
                    if (options_.lambda) {
                        data = eve::fnma(eve::wide<float>(options_.lambda), data, data);
                    }

                    eve::store(data, &datas[i * CHUNK_SIZE + OFFSET]);
                });
            }

            for (const auto& i : std::views::iota(0, REM)) {
                const auto idx = i + BLOCKED;
                auto grad = [&]() noexcept -> float {
                    auto grad = grads[idx];
                    if (options_.minimize) {
                        grad = -grad;
                    }

                    auto mom = options_.beta1 * moms[idx] + (1 - options_.beta1) * grad;
                    auto sqr = (options_.adabelief) ? (grad - mom) * (grad - mom) + options_.eps
                                                    : grad * grad;
                    auto vel = options_.beta2 * vels[idx] + (1 - options_.beta2) * sqr;

                    mom /= (1 - std::pow(options_.beta1, state_.step));
                    vel /= (1 - std::pow(options_.beta2, state_.step));

                    auto upd = mom / (std::sqrt(vel) + options_.eps);
                    if (options_.cautious) {
                        auto grad_sign = (grad > 0);
                        auto upd_sign = (upd > 0);
                        upd = (grad_sign == upd_sign) ? upd : 0;
                    }

                    moms[idx] = mom;
                    vels[idx] = vel;

                    return upd;
                }();

                auto data = datas[idx];
                datas[idx] = data * options_.lr * grad;
                if (options_.lambda) {
                    datas[idx] -= options_.lambda * data;
                }
            }
        }
    }

    private:
    AdamOptions options_;
    detail::AdamState state_;
    std::vector<std::reference_wrapper<Tunable>> tunables_;
};
}  // namespace optim
}  // namespace tune
