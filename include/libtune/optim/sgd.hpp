#pragma once

#include <eve/module/core.hpp>
#include <ranges>
#include <type_traits>
#include <vector>

#include "../detail/utils.hpp"
#include "../tunable.hpp"

namespace tune {
namespace detail {
struct SGDState {
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    SGDState(T&... tunables) {
        (momentum_.emplace_back(std::vector<float>(tunables.numel())), ...);
    }

    std::vector<std::vector<float>> momentum_{};
    int step = 0;
};
}  // namespace detail

namespace optim {
struct SGDOptions {
    float lr = 0.01;
    float momentum = 0.9;
    float lambda = 0.0;

    bool nesterov = true;
    bool minimize = false;
    bool cautious = false;
};

class SGD {
    public:
    template <typename... T>
        requires(std::is_same_v<T, Tunable> && ...)
    SGD(SGDOptions options, T&... tunables) : options_(options), state_(tunables...) {
        (tunables_.emplace_back(tunables), ...);
    }

    auto zero_grad() -> void {
        for (auto& tnbl : tunables_) {
            auto t = tnbl.get();
            t.zero_grad();
        }
    }

    auto step() -> void {
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
                    auto grad = [&]() -> eve::wide<float> {
                        auto grad = eve::wide<float>(&grads[i * CHUNK_SIZE + OFFSET]);
                        if (options_.minimize) {
                            grad = -grad;
                        }

                        if (options_.momentum) {
                            auto mom = eve::wide<float>(&moms[i * CHUNK_SIZE + OFFSET]);
                            mom = eve::fma(eve::wide<float>(options_.momentum), mom, grad);
                            eve::store(mom, &moms[i * CHUNK_SIZE + OFFSET]);

                            auto upd =
                                (options_.nesterov)
                                    ? eve::fma(eve::wide<float>(options_.momentum), mom, grad)
                                    : mom;

                            if (options_.cautious) {
                                auto grad_sign = eve::sign(grad);
                                auto upd_sign = eve::sign(upd);
                                auto mask = (grad_sign == upd_sign);
                                upd = eve::if_else(mask, upd, eve::zero);
                            }
                            grad = upd;
                        }

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
                auto grad = [&]() -> float {
                    auto grad = grads[idx];
                    if (options_.minimize) {
                        grad = -grad;
                    }

                    if (options_.momentum) {
                        auto mom = moms[idx] * options_.momentum + grad;
                        moms[idx] = mom;

                        auto upd = (options_.nesterov) ? mom * options_.momentum + grad : mom;
                        if (options_.cautious) {
                            auto grad_sign = (grad > 0);
                            auto upd_sign = (upd > 0);
                            upd = (grad_sign == upd_sign) ? upd : 0;
                        }
                        grad = upd;
                    }
                    return grad;
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
    SGDOptions options_;
    detail::SGDState state_;
    std::vector<std::reference_wrapper<Tunable>> tunables_;
};
}  // namespace optim
}  // namespace tune
