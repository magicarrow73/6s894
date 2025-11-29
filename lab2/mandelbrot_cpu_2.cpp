// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;
// use this number of parallel executions given by ILP at any given time.
// assume that img_size is a multiple of (16 * ILP_SCALING).
// otherwise we will need boundary checks.
constexpr uint32_t ILP_SCALING = 4;
constexpr uint32_t num_cores = 8;            // for multicore
constexpr uint32_t num_threads_per_core = 4; // for multicore + multithread

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->

// OPTIONAL: Uncomment this block to include your CPU vector implementation
// from Lab 1 for easy comparison.
//
// (If you do this, you'll need to update your code to use the new constants
// 'window_zoom', 'window_x', and 'window_y'.)

#define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector

void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // constants vectorized
    __m512 img_size_vec = _mm512_set1_ps(float(img_size));
    __m512 four = _mm512_set1_ps(4.0f);
    __m512i one_epi = _mm512_set1_epi32(1);
    __m512i max_iters_epi = _mm512_set1_epi32(int32_t(max_iters));
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; j += 16) {
            // set float(j) vector
            __m512 j_vec = _mm512_set_ps(
                float(j + 15),
                float(j + 14),
                float(j + 13),
                float(j + 12),
                float(j + 11),
                float(j + 10),
                float(j + 9),
                float(j + 8),
                float(j + 7),
                float(j + 6),
                float(j + 5),
                float(j + 4),
                float(j + 3),
                float(j + 2),
                float(j + 1),
                float(j + 0));

            // set float(i) vector
            __m512 i_vec = _mm512_set1_ps(float(i));

            // get coordinates cx, cy
            __m512 cx = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(j_vec, img_size_vec), window_zoom_vec),
                window_x_vec);
            __m512 cy = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(i_vec, img_size_vec), window_zoom_vec),
                window_y_vec);

            // init x2,y2,w vectors
            __m512 x2 = _mm512_set1_ps(0.0f);
            __m512 y2 = _mm512_set1_ps(0.0f);
            __m512 w = _mm512_set1_ps(0.0f);
            __m512i iters = _mm512_set1_epi32(0);
            // mask
            __mmask16 mask = _mm512_cmp_ps_mask(_mm512_add_ps(x2, y2), four, _CMP_LE_OQ);

            while (mask != 0) {
                // update mask
                mask = _mm512_cmp_ps_mask(_mm512_add_ps(x2, y2), four, _CMP_LE_OQ);
                __mmask16 iters_mask =
                    _mm512_cmp_epi32_mask(iters, max_iters_epi, _MM_CMPINT_LT);
                mask = _mm512_kand(mask, iters_mask);
                // compute x,y vectors

                __m512 x = _mm512_add_ps(_mm512_sub_ps(x2, y2), cx);
                __m512 y = _mm512_add_ps(_mm512_sub_ps(w, _mm512_add_ps(x2, y2)), cy);
                // compute x2, y2, w; these should not be updated if we have already
                // escaped (hence we multiply w/ mask)
                x2 = _mm512_mask_mul_ps(x2, mask, x, x);
                y2 = _mm512_mask_mul_ps(y2, mask, y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mask_mul_ps(w, mask, z, z);

                // update iters
                iters = _mm512_mask_add_epi32(iters, mask, iters, one_epi);
            }
            // store result
            _mm512_storeu_si512((void *)(out + i * img_size + j), iters);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // constants vectorized
    __m512 img_size_vec = _mm512_set1_ps(float(img_size));
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    __m512 four = _mm512_set1_ps(4.0f);
    __m512i one_epi = _mm512_set1_epi32(1);
    __m512i max_iters_epi = _mm512_set1_epi32(int32_t(max_iters));
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; j += 16 * ILP_SCALING) {
            // set float(j) vectors
            __m512 j_vecs[ILP_SCALING];

#pragma unroll
            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                j_vecs[ilp] = _mm512_set_ps(
                    float(j + ilp * 16 + 15),
                    float(j + ilp * 16 + 14),
                    float(j + ilp * 16 + 13),
                    float(j + ilp * 16 + 12),
                    float(j + ilp * 16 + 11),
                    float(j + ilp * 16 + 10),
                    float(j + ilp * 16 + 9),
                    float(j + ilp * 16 + 8),
                    float(j + ilp * 16 + 7),
                    float(j + ilp * 16 + 6),
                    float(j + ilp * 16 + 5),
                    float(j + ilp * 16 + 4),
                    float(j + ilp * 16 + 3),
                    float(j + ilp * 16 + 2),
                    float(j + ilp * 16 + 1),
                    float(j + ilp * 16 + 0));
            }
            // set float(i) vectors
            __m512 i_vecs[ILP_SCALING];
            // set coordinates cx, cy vectors
            __m512 cx_vecs[ILP_SCALING];
            __m512 cy_vecs[ILP_SCALING];

            // init x2, y2, w
            __m512 x2_vecs[ILP_SCALING];
            __m512 y2_vecs[ILP_SCALING];
            __m512 w_vecs[ILP_SCALING];
            // init iters_vec, mask_vecs
            __m512i iters_vec[ILP_SCALING];
            __mmask16 mask_vecs[ILP_SCALING];

#pragma unroll
            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                i_vecs[ilp] = _mm512_set1_ps(float(i));

                __m512 j_vec = j_vecs[ilp];
                __m512 i_vec = i_vecs[ilp];
                cx_vecs[ilp] = _mm512_add_ps(
                    _mm512_mul_ps(_mm512_div_ps(j_vec, img_size_vec), window_zoom_vec),
                    window_x_vec);
                cy_vecs[ilp] = _mm512_add_ps(
                    _mm512_mul_ps(_mm512_div_ps(i_vec, img_size_vec), window_zoom_vec),
                    window_y_vec);

                x2_vecs[ilp] = _mm512_set1_ps(0.0f);
                y2_vecs[ilp] = _mm512_set1_ps(0.0f);
                w_vecs[ilp] = _mm512_set1_ps(0.0f);
                iters_vec[ilp] = _mm512_set1_epi32(0);

                mask_vecs[ilp] = 0xFFFF;
            }
            // combined_or to check while loop condition, for simplicity
            __mmask16 combined_or = 0;
#pragma unroll
            for (int k = 0; k < ILP_SCALING; ++k) {
                combined_or |= mask_vecs[k]; // bitwise OR
            }

            __m512 x_vecs[ILP_SCALING];
            __m512 y_vecs[ILP_SCALING];
            __mmask16 iters_mask[ILP_SCALING];
            // while loop
            while (combined_or != 0) {

#pragma unroll
                for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {

                    // first, update mask vector
                    mask_vecs[ilp] = _mm512_cmp_ps_mask(
                        _mm512_add_ps(x2_vecs[ilp], y2_vecs[ilp]),
                        four,
                        _CMP_LE_OQ);

                    // also check num_iters
                    iters_mask[ilp] = _mm512_cmp_epi32_mask(
                        iters_vec[ilp],
                        max_iters_epi,
                        _MM_CMPINT_LT);
                    mask_vecs[ilp] = _mm512_kand(mask_vecs[ilp], iters_mask[ilp]);

                    // now compute the vectors for those lanes that are still active (i.e.
                    // have not escaped and have not reached #iters >= max_iters) compute
                    // x,y vectors
                    x_vecs[ilp] = _mm512_add_ps(
                        _mm512_sub_ps(x2_vecs[ilp], y2_vecs[ilp]),
                        cx_vecs[ilp]);
                    y_vecs[ilp] = _mm512_add_ps(
                        _mm512_sub_ps(
                            w_vecs[ilp],
                            _mm512_add_ps(x2_vecs[ilp], y2_vecs[ilp])),
                        cy_vecs[ilp]);

                    // now compute x2, y2, w vectors

                    x2_vecs[ilp] = _mm512_mask_mul_ps(
                        x2_vecs[ilp],
                        mask_vecs[ilp],
                        x_vecs[ilp],
                        x_vecs[ilp]);
                    y2_vecs[ilp] = _mm512_mask_mul_ps(
                        y2_vecs[ilp],
                        mask_vecs[ilp],
                        y_vecs[ilp],
                        y_vecs[ilp]);
                    __m512 z_vec = _mm512_add_ps(x_vecs[ilp], y_vecs[ilp]);
                    w_vecs[ilp] =
                        _mm512_mask_mul_ps(w_vecs[ilp], mask_vecs[ilp], z_vec, z_vec);

                    // update total #iters
                    iters_vec[ilp] = _mm512_mask_add_epi32(
                        iters_vec[ilp],
                        mask_vecs[ilp],
                        iters_vec[ilp],
                        one_epi);
                }

                combined_or = 0;
#pragma unroll
                for (int k = 0; k < ILP_SCALING; ++k) {
                    combined_or |= mask_vecs[k]; // bitwise OR
                }
            }
            // store results

            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                uint32_t offset = ilp * 16;
                _mm512_storeu_si512(
                    (void *)(out + i * img_size + j + offset),
                    iters_vec[ilp]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

// thread data + fns

// struct to hold thread data to pass as argument
struct ThreadData {
    uint32_t thread_id;
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
    uint32_t total_threads;
};
// thread function to execute mandelbrot computation
void *thread_func(void *arg) {
    ThreadData *data = (ThreadData *)(arg);
    uint32_t thread_id = data->thread_id;
    uint32_t img_size = data->img_size;
    uint32_t max_iters = data->max_iters;
    uint32_t *out = data->out;
    uint32_t total_threads = data->total_threads; // for multicore

    // constants vectorized
    // constants vectorized
    __m512 img_size_vec = _mm512_set1_ps(float(img_size));
    __m512 four = _mm512_set1_ps(4.0f);
    __m512i one_epi = _mm512_set1_epi32(1);
    __m512i max_iters_epi = _mm512_set1_epi32(int32_t(max_iters));
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    for (uint64_t i = thread_id; i < img_size; i += total_threads) {
        for (uint64_t j = 0; j < img_size; j += 16) {
            // set float(j) vector
            __m512 j_vec = _mm512_set_ps(
                float(j + 15),
                float(j + 14),
                float(j + 13),
                float(j + 12),
                float(j + 11),
                float(j + 10),
                float(j + 9),
                float(j + 8),
                float(j + 7),
                float(j + 6),
                float(j + 5),
                float(j + 4),
                float(j + 3),
                float(j + 2),
                float(j + 1),
                float(j + 0));

            // set float(i) vector
            __m512 i_vec = _mm512_set1_ps(float(i));

            // get coordinates cx, cy
            __m512 cx = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(j_vec, img_size_vec), window_zoom_vec),
                window_x_vec);
            __m512 cy = _mm512_add_ps(
                _mm512_mul_ps(_mm512_div_ps(i_vec, img_size_vec), window_zoom_vec),
                window_y_vec);

            // init x2,y2,w vectors
            __m512 x2 = _mm512_set1_ps(0.0f);
            __m512 y2 = _mm512_set1_ps(0.0f);
            __m512 w = _mm512_set1_ps(0.0f);
            __m512i iters = _mm512_set1_epi32(0);
            // mask
            __mmask16 mask = _mm512_cmp_ps_mask(_mm512_add_ps(x2, y2), four, _CMP_LE_OQ);

            while (mask != 0) {
                // update mask
                mask = _mm512_cmp_ps_mask(_mm512_add_ps(x2, y2), four, _CMP_LE_OQ);
                __mmask16 iters_mask =
                    _mm512_cmp_epi32_mask(iters, max_iters_epi, _MM_CMPINT_LT);
                mask = _mm512_kand(mask, iters_mask);
                // compute x,y vectors

                __m512 x = _mm512_add_ps(_mm512_sub_ps(x2, y2), cx);
                __m512 y = _mm512_add_ps(_mm512_sub_ps(w, _mm512_add_ps(x2, y2)), cy);

                _mm512_sub_ps(_mm512_add_ps(w, cy), _mm512_add_ps(x2, y2));
                // compute x2, y2, w; these should not be updated if we have already
                // escaped (hence we multiply w/ mask)
                x2 = _mm512_mask_mul_ps(x2, mask, x, x);
                y2 = _mm512_mask_mul_ps(y2, mask, y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mask_mul_ps(w, mask, z, z);

                // update iters
                iters = _mm512_mask_add_epi32(iters, mask, iters, one_epi);
            }
            // store result
            _mm512_storeu_si512((void *)(out + i * img_size + j), iters);
        }
    }
    return nullptr;
}

void mandelbrot_cpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {

    // spawn threads to compute different rows in parallel
    // create threads

    pthread_t threads[num_cores];

    ThreadData thread_data[num_cores];
    for (uint32_t t = 0; t < num_cores; t++) {
        thread_data[t].total_threads = num_cores;
    }

    // create + join threads
    for (uint32_t t = 0; t < num_cores; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        pthread_create(&threads[t], nullptr, thread_func, (void *)(&thread_data[t]));
    }

    for (uint32_t t = 0; t < num_cores; t++) {
        pthread_join(threads[t], nullptr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multicore_multithread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    pthread_t threads[num_cores * num_threads_per_core];

    ThreadData thread_data[num_cores * num_threads_per_core];
    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        thread_data[t].total_threads = num_cores * num_threads_per_core;
    }

    // create + join threads
    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        pthread_create(&threads[t], nullptr, thread_func, (void *)(&thread_data[t]));
    }

    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        pthread_join(threads[t], nullptr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

void *thread_func_ilp(void *arg) {

    ThreadData *data = (ThreadData *)(arg);
    uint32_t thread_id = data->thread_id;
    uint32_t img_size = data->img_size;
    uint32_t max_iters = data->max_iters;
    uint32_t *out = data->out;
    uint32_t total_threads = data->total_threads; // for multicore + multithread
    // constants vectorized
    __m512 img_size_vec = _mm512_set1_ps(float(img_size));
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    __m512 four = _mm512_set1_ps(4.0f);
    __m512i one_epi = _mm512_set1_epi32(1);
    __m512i max_iters_epi = _mm512_set1_epi32(int32_t(max_iters));
    for (uint64_t i = thread_id; i < img_size; i += total_threads) {
        for (uint64_t j = 0; j < img_size; j += 16 * ILP_SCALING) {
            // set float(j) vectors
            __m512 j_vecs[ILP_SCALING];

#pragma unroll
            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                j_vecs[ilp] = _mm512_set_ps(
                    float(j + ilp * 16 + 15),
                    float(j + ilp * 16 + 14),
                    float(j + ilp * 16 + 13),
                    float(j + ilp * 16 + 12),
                    float(j + ilp * 16 + 11),
                    float(j + ilp * 16 + 10),
                    float(j + ilp * 16 + 9),
                    float(j + ilp * 16 + 8),
                    float(j + ilp * 16 + 7),
                    float(j + ilp * 16 + 6),
                    float(j + ilp * 16 + 5),
                    float(j + ilp * 16 + 4),
                    float(j + ilp * 16 + 3),
                    float(j + ilp * 16 + 2),
                    float(j + ilp * 16 + 1),
                    float(j + ilp * 16 + 0));
            }
            // set float(i) vectors
            __m512 i_vecs[ILP_SCALING];
            // set coordinates cx, cy vectors
            __m512 cx_vecs[ILP_SCALING];
            __m512 cy_vecs[ILP_SCALING];

            // init x2, y2, w
            __m512 x2_vecs[ILP_SCALING];
            __m512 y2_vecs[ILP_SCALING];
            __m512 w_vecs[ILP_SCALING];
            // init iters_vec, mask_vecs
            __m512i iters_vec[ILP_SCALING];
            __mmask16 mask_vecs[ILP_SCALING];

#pragma unroll
            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                i_vecs[ilp] = _mm512_set1_ps(float(i));

                __m512 j_vec = j_vecs[ilp];
                __m512 i_vec = i_vecs[ilp];
                cx_vecs[ilp] = _mm512_add_ps(
                    _mm512_mul_ps(_mm512_div_ps(j_vec, img_size_vec), window_zoom_vec),
                    window_x_vec);
                cy_vecs[ilp] = _mm512_add_ps(
                    _mm512_mul_ps(_mm512_div_ps(i_vec, img_size_vec), window_zoom_vec),
                    window_y_vec);

                x2_vecs[ilp] = _mm512_set1_ps(0.0f);
                y2_vecs[ilp] = _mm512_set1_ps(0.0f);
                w_vecs[ilp] = _mm512_set1_ps(0.0f);
                iters_vec[ilp] = _mm512_set1_epi32(0);

                mask_vecs[ilp] = 0xFFFF;
            }
            // combined_or to check while loop condition, for simplicity
            __mmask16 combined_or = 0;
#pragma unroll
            for (int k = 0; k < ILP_SCALING; ++k) {
                combined_or |= mask_vecs[k]; // bitwise OR
            }

            __m512 x_vecs[ILP_SCALING];
            __m512 y_vecs[ILP_SCALING];
            __mmask16 iters_mask[ILP_SCALING];
            // while loop
            while (combined_or != 0) {

#pragma unroll
                for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {

                    // first, update mask vector
                    mask_vecs[ilp] = _mm512_cmp_ps_mask(
                        _mm512_add_ps(x2_vecs[ilp], y2_vecs[ilp]),
                        four,
                        _CMP_LE_OQ);

                    // also check num_iters
                    iters_mask[ilp] = _mm512_cmp_epi32_mask(
                        iters_vec[ilp],
                        max_iters_epi,
                        _MM_CMPINT_LT);
                    mask_vecs[ilp] = _mm512_kand(mask_vecs[ilp], iters_mask[ilp]);

                    // now compute the vectors for those lanes that are still active (i.e.
                    // have not escaped and have not reached #iters >= max_iters) compute
                    // x,y vectors
                    x_vecs[ilp] = _mm512_add_ps(
                        _mm512_sub_ps(x2_vecs[ilp], y2_vecs[ilp]),
                        cx_vecs[ilp]);
                    y_vecs[ilp] = _mm512_add_ps(
                        _mm512_sub_ps(
                            w_vecs[ilp],
                            _mm512_add_ps(x2_vecs[ilp], y2_vecs[ilp])),
                        cy_vecs[ilp]);

                    // now compute x2, y2, w vectors

                    x2_vecs[ilp] = _mm512_mask_mul_ps(
                        x2_vecs[ilp],
                        mask_vecs[ilp],
                        x_vecs[ilp],
                        x_vecs[ilp]);
                    y2_vecs[ilp] = _mm512_mask_mul_ps(
                        y2_vecs[ilp],
                        mask_vecs[ilp],
                        y_vecs[ilp],
                        y_vecs[ilp]);
                    __m512 z_vec = _mm512_add_ps(x_vecs[ilp], y_vecs[ilp]);
                    w_vecs[ilp] =
                        _mm512_mask_mul_ps(w_vecs[ilp], mask_vecs[ilp], z_vec, z_vec);

                    // update total #iters
                    iters_vec[ilp] = _mm512_mask_add_epi32(
                        iters_vec[ilp],
                        mask_vecs[ilp],
                        iters_vec[ilp],
                        one_epi);
                }

                combined_or = 0;
#pragma unroll
                for (int k = 0; k < ILP_SCALING; ++k) {
                    combined_or |= mask_vecs[k]; // bitwise OR
                }
            }
            // store results

            for (uint32_t ilp = 0; ilp < ILP_SCALING; ilp++) {
                uint32_t offset = ilp * 16;
                _mm512_storeu_si512(
                    (void *)(out + i * img_size + j + offset),
                    iters_vec[ilp]);
            }
        }
    }
    return nullptr;
}
void mandelbrot_cpu_vector_multicore_multithread_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {

    pthread_t threads[num_cores * num_threads_per_core];
    ThreadData thread_data[num_cores * num_threads_per_core];
    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        thread_data[t].total_threads = num_cores * num_threads_per_core;
    }

    // create + join threads
    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        pthread_create(&threads[t], nullptr, thread_func_ilp, (void *)(&thread_data[t]));
    }
    for (uint32_t t = 0; t < num_cores * num_threads_per_core; t++) {
        pthread_join(threads[t], nullptr);
    }
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
