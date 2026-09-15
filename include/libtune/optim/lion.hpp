#pragma once

#include <eve/module/core.hpp>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../detail/utils.hpp"
#include "../tunable.hpp"

namespace tune {
namespace detail {
struct LionState {
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    LionState(T&... tunables) {
        (momentum_.emplace_back(std::vector<float>(tunables.numel())), ...);
    }

    std::vector<std::vector<float>> momentum_{};
    int step = 0;
};
}  // namespace detail

namespace optim {
struct LionOptions {
    float lr = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float lambda = 0.0f;

    bool minimize = false;
    bool cautious = false;
    bool adabelief = false;
};

class Lion {
    public:
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    Lion(const LionOptions& options, T&... tunables) : options_(options), state_(tunables...) {
        (tunables_.emplace_back(tunables), ...);
    }

    constexpr auto zero_grad() noexcept -> void {
        for (auto& tnbl : tunables_) {
            auto t = tnbl.get();
            t.zero_grad();
        }
    }

    // TODO: also also noexcept this
    constexpr auto step() -> void {
        for (auto&& [index, tnbl] : std::views::enumerate(tunables_)) {
            auto& t = tnbl.get();
            auto& moms = state_.momentum_[index];
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
                        auto pre = eve::fma(beta1, mom, grad);
                        pre = eve::fnma(beta1, grad, pre);

                        auto upd = eve::sign(pre);
                        if (options_.cautious) {
                            auto grad_sign = (grad > 0);
                            auto upd_sign = (upd > 0);
                            auto mask = (grad_sign == upd_sign);
                            upd = eve::if_else(mask, upd, eve::zero);
                        }

                        mom = eve::fma(beta2, mom, grad);
                        mom = eve::fnma(beta2, grad, mom);
                        eve::store(mom, &moms[i * CHUNK_SIZE + OFFSET]);

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

                    auto pre = options_.beta1 * moms[idx] + (1 - options_.beta1) * grad;
                    auto upd = (pre > 0) ? 1 : (pre < 0) ? -1 : 0;
                    if (options_.cautious) {
                        auto grad_sign = (grad > 0);
                        auto upd_sign = (upd > 0);
                        upd = (grad_sign == upd_sign) ? upd : 0;
                    }

                    auto mom = options_.beta2 * moms[idx] + (1 - options_.beta2) * grad;
                    moms[idx] = mom;

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
    LionOptions options_;
    detail::LionState state_;
    std::vector<std::reference_wrapper<Tunable>> tunables_;
};
}  // namespace optim
}  // namespace tune
