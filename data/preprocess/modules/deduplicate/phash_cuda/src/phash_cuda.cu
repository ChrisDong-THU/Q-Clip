#include "phash_cuda.hpp"

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline void throwIf(bool cond, const char* msg) {
    if (cond) {
        throw std::runtime_error(msg);
    }
}

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline void nvjpegCheck(nvjpegStatus_t st, const char* msg) {
    if (st != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": nvjpeg status " + std::to_string(static_cast<int>(st)));
    }
}

template <class T>
class WorkQueue {
public:
    explicit WorkQueue(size_t capacity) : m_capacity(capacity) {}

    void push(const T& item) {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cvNotFull.wait(lock, [&]() { return m_queue.size() < m_capacity; });
            m_queue.push_back(item);
        }
        m_cvNotEmpty.notify_one();
    }

    void pushMany(const std::vector<T>& items) {
        for (const auto& item : items) {
            push(item);
        }
    }

    std::vector<T> popMax(size_t maxCount) {
        std::vector<T> out;
        out.reserve(maxCount);

        std::unique_lock<std::mutex> lock(m_mutex);
        m_cvNotEmpty.wait(lock, [&]() { return m_sentinel || !m_queue.empty(); });
        if (m_queue.empty()) {
            return out;
        }

        size_t take = std::min(maxCount, m_queue.size());
        for (size_t i = 0; i < take; ++i) {
            out.push_back(m_queue.front());
            m_queue.pop_front();
        }
        lock.unlock();
        m_cvNotFull.notify_all();
        return out;
    }

    void setSentinel() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_sentinel = true;
        }
        m_cvNotEmpty.notify_all();
    }

    bool isSentinel() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_sentinel;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }

private:
    size_t m_capacity;
    mutable std::mutex m_mutex;
    std::condition_variable m_cvNotEmpty;
    std::condition_variable m_cvNotFull;
    bool m_sentinel{ false };
    std::deque<T> m_queue;
};

struct GpuData {
    const unsigned char* decodedPtr{ nullptr };
    float* resizedPtr{ nullptr };
    int originalWidth{ 0 };
    int originalHeight{ 0 };
};

struct ImageTask {
    int index{ -1 };
    std::string path;

    unsigned char* h_data{ nullptr };
    size_t h_size{ 0 };
    int width{ 0 };
    int height{ 0 };

    unsigned char* d_decoded{ nullptr };
    float* d_resized{ nullptr };
};

__device__ inline int clampInt(int x, int minV, int maxV) {
    return x < minV ? minV : (x > maxV ? maxV : x);
}

__device__ __forceinline__ float sincpi(float x) {
    float ax = fabsf(x);
    if (ax < 1e-6f) {
        return 1.0f;
    }
    float pix = kPi * x;
    return __sinf(pix) / pix;
}

__device__ __forceinline__ float lanczos3Weight(float x) {
    float ax = fabsf(x);
    if (ax >= 3.0f) {
        return 0.0f;
    }
    return sincpi(x) * sincpi(x / 3.0f);
}

__global__ void batchLanczosResizeKernel(const GpuData* data, int outSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int imgIdx = blockIdx.z;

    if (x >= outSize || y >= outSize) {
        return;
    }

    const unsigned char* d_in = data[imgIdx].decodedPtr;
    float* d_out = data[imgIdx].resizedPtr;
    int inWidth = data[imgIdx].originalWidth;
    int inHeight = data[imgIdx].originalHeight;
    if (inWidth <= 0 || inHeight <= 0) {
        return;
    }

    float scaleX = static_cast<float>(inWidth) / static_cast<float>(outSize);
    float scaleY = static_cast<float>(inHeight) / static_cast<float>(outSize);
    float u = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    float v = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

    int xCenter = static_cast<int>(floorf(u));
    int yCenter = static_cast<int>(floorf(v));

    float sx = (scaleX > 1.0f) ? scaleX : 1.0f;
    float sy = (scaleY > 1.0f) ? scaleY : 1.0f;
    int radiusX = static_cast<int>(ceilf(3.0f * sx));
    int radiusY = static_cast<int>(ceilf(3.0f * sy));
    radiusX = min(radiusX, 64);
    radiusY = min(radiusY, 64);

    float sum = 0.0f;
    float totalW = 0.0f;

    for (int j = -radiusY; j <= radiusY; ++j) {
        int yy = clampInt(yCenter + j, 0, inHeight - 1);
        float wy = lanczos3Weight((v - static_cast<float>(yCenter + j)) / sy);
        if (wy == 0.0f) {
            continue;
        }
        for (int i = -radiusX; i <= radiusX; ++i) {
            int xx = clampInt(xCenter + i, 0, inWidth - 1);
            float wx = lanczos3Weight((u - static_cast<float>(xCenter + i)) / sx);
            float w = wx * wy;
            if (w == 0.0f) {
                continue;
            }
            int base = (yy * inWidth + xx) * 3;
            float r = static_cast<float>(d_in[base + 0]);
            float g = static_cast<float>(d_in[base + 1]);
            float b = static_cast<float>(d_in[base + 2]);
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            sum += w * gray;
            totalW += w;
        }
    }

    if (totalW != 0.0f) {
        sum /= totalW;
    }

    d_out[y * outSize + x] = sum;
}

__device__ __forceinline__ float posInf() {
    return __int_as_float(0x7f800000);
}

__global__ void phashDct8x8Kernel(
    const float* const* __restrict__ d_imgs,
    int N,
    int K,
    const int* __restrict__ d_outPos,
    uint8_t* __restrict__ d_outBytes,
    int batchSize
) {
    int imgIdx = blockIdx.x;
    if (imgIdx >= batchSize) {
        return;
    }

    int outIndex = d_outPos[imgIdx];
    const float* img = d_imgs[imgIdx];

    int tid = threadIdx.x;
    if (tid >= 64) {
        return;
    }

    int u = tid / 8;
    int v = tid % 8;

    float coeff = 0.0f;
    if (u < K && v < K) {
        for (int y = 0; y < N; ++y) {
            float cy = __cosf((static_cast<float>(kPi) * static_cast<float>(v) * (2.0f * static_cast<float>(y) + 1.0f)) / (2.0f * static_cast<float>(N)));
            const float* row = img + y * N;
            for (int x = 0; x < N; ++x) {
                float cx = __cosf((static_cast<float>(kPi) * static_cast<float>(u) * (2.0f * static_cast<float>(x) + 1.0f)) / (2.0f * static_cast<float>(N)));
                coeff += row[x] * cx * cy;
            }
        }
    }

    __shared__ float sCoeff[64];
    __shared__ float sSort[64];
    sCoeff[tid] = coeff;

    float sortVal = coeff;
    int pixels = K * K;
    if (tid == 0 || tid >= pixels) {
        sortVal = posInf();
    }
    sSort[tid] = sortVal;
    __syncthreads();

    for (int k = 2; k <= 64; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                float a = sSort[tid];
                float b = sSort[ixj];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    sSort[tid] = b;
                    sSort[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int n = pixels - 1;
        float median = 0.0f;
        if (n <= 0) {
            median = 0.0f;
        } else if (n & 1) {
            median = sSort[n / 2];
        } else {
            median = 0.5f * (sSort[n / 2 - 1] + sSort[n / 2]);
        }
        uint8_t* out = d_outBytes + static_cast<size_t>(outIndex) * 8;
        for (int i = 0; i < 8; ++i) {
            out[i] = 0;
        }
        for (int i = 0; i < 64; ++i) {
            if (i >= pixels) {
                continue;
            }
            if (sCoeff[i] >= median) {
                int byteIndex = i / 8;
                int bitInByte = 7 - (i % 8);
                out[byteIndex] = static_cast<uint8_t>(out[byteIndex] | static_cast<uint8_t>(1u << bitInByte));
            }
        }
    }
}

class CudaPhashPipeline {
public:
    CudaPhashPipeline(int resize, int lowFreq, int batchSize, int readerThreads)
        : m_resize(resize),
          m_lowFreq(lowFreq),
          m_batchSize(batchSize),
          m_readerThreads(readerThreads),
          m_inputQueue(static_cast<size_t>(batchSize) * 8),
          m_decodeQueue(static_cast<size_t>(batchSize) * 8),
          m_resizeQueue(static_cast<size_t>(batchSize) * 8) {
        throwIf(m_resize <= 0, "resize must be > 0");
        throwIf(m_resize > 1024, "resize too large");
        throwIf(m_lowFreq <= 0, "low_freq must be > 0");
        throwIf(m_batchSize <= 0, "batch_size must be > 0");

        nvjpegCheck(nvjpegCreateSimple(&m_nvjHandle), "nvjpegCreateSimple failed");
        nvjpegCheck(nvjpegJpegStateCreate(m_nvjHandle, &m_nvjDecoderState), "nvjpegJpegStateCreate failed");
        nvjpegCheck(nvjpegJpegStateCreate(m_nvjHandle, &m_nvjSingleDecoderState), "nvjpegJpegStateCreate(single) failed");

        cudaCheck(cudaStreamCreate(&m_decodeStream), "cudaStreamCreate(decode) failed");
        cudaCheck(cudaStreamCreate(&m_resizeStream), "cudaStreamCreate(resize) failed");
        cudaCheck(cudaStreamCreate(&m_hashStream), "cudaStreamCreate(hash) failed");

        cudaCheck(cudaMalloc(&m_d_AArray, m_batchSize * sizeof(const float*)), "cudaMalloc(d_AArray) failed");
        cudaCheck(cudaMalloc(&m_d_outPos, m_batchSize * sizeof(int)), "cudaMalloc(d_outPos) failed");
        cudaCheck(cudaMalloc(&m_d_gpuDataVec, m_batchSize * sizeof(GpuData)), "cudaMalloc(d_gpuDataVec) failed");
    }

    ~CudaPhashPipeline() {
        if (m_d_gpuDataVec) cudaFree(m_d_gpuDataVec);
        if (m_d_outPos) cudaFree(m_d_outPos);
        if (m_d_AArray) cudaFree(m_d_AArray);

        if (m_decodeStream) cudaStreamDestroy(m_decodeStream);
        if (m_resizeStream) cudaStreamDestroy(m_resizeStream);
        if (m_hashStream) cudaStreamDestroy(m_hashStream);

        if (m_nvjDecoderState) nvjpegJpegStateDestroy(m_nvjDecoderState);
        if (m_nvjSingleDecoderState) nvjpegJpegStateDestroy(m_nvjSingleDecoderState);
        if (m_nvjHandle) nvjpegDestroy(m_nvjHandle);
    }

    std::vector<std::array<uint8_t, 8>> computeBytes(const std::vector<std::string>& paths) {
        if (paths.empty()) {
            return {};
        }

        std::vector<ImageTask> tasks(paths.size());
        for (size_t i = 0; i < paths.size(); ++i) {
            tasks[i].index = static_cast<int>(i);
            tasks[i].path = paths[i];
        }

        uint8_t* d_hashBytes = nullptr;
        cudaCheck(cudaMalloc(&d_hashBytes, tasks.size() * 8), "cudaMalloc(d_hashBytes) failed");
        cudaCheck(cudaMemset(d_hashBytes, 0, tasks.size() * 8), "cudaMemset(d_hashBytes) failed");

        WorkQueue<ImageTask*> inputQueue(tasks.size());
        for (auto& t : tasks) {
            inputQueue.push(&t);
        }
        inputQueue.setSentinel();

        std::vector<std::thread> readers;
        readers.reserve(static_cast<size_t>(m_readerThreads));
        for (int i = 0; i < m_readerThreads; ++i) {
            readers.emplace_back([&]() { reader(inputQueue, m_decodeQueue); });
        }

        std::thread decoderThread([&]() { decoder(m_decodeQueue, m_resizeQueue); });
        std::thread resizerThread([&]() { resizer(m_resizeQueue, m_inputQueue); });
        std::thread hasherThread([&]() { hasher(m_inputQueue, d_hashBytes); });

        for (auto& t : readers) {
            t.join();
        }
        m_decodeQueue.setSentinel();

        decoderThread.join();
        resizerThread.join();
        hasherThread.join();

        std::vector<std::array<uint8_t, 8>> h_hashes(tasks.size());
        cudaCheck(cudaMemcpy(h_hashes.data(), d_hashBytes, tasks.size() * 8, cudaMemcpyDeviceToHost), "cudaMemcpy hashes failed");
        cudaFree(d_hashBytes);
        return h_hashes;
    }

private:
    void reader(WorkQueue<ImageTask*>& inQueue, WorkQueue<ImageTask*>& outQueue) {
        while (true) {
            auto batch = inQueue.popMax(1);
            if (batch.empty()) {
                return;
            }
            ImageTask* task = batch[0];

            std::ifstream file(task->path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                outQueue.push(task);
                continue;
            }
            std::streamsize sz = file.tellg();
            if (sz <= 0) {
                outQueue.push(task);
                continue;
            }
            task->h_size = static_cast<size_t>(sz);

            unsigned char* h_data = nullptr;
            if (cudaHostAlloc(reinterpret_cast<void**>(&h_data), task->h_size, cudaHostAllocDefault) == cudaSuccess) {
                task->h_data = h_data;
            } else {
                task->h_data = static_cast<unsigned char*>(malloc(task->h_size));
            }

            file.seekg(0, std::ios::beg);
            file.read(reinterpret_cast<char*>(task->h_data), sz);
            file.close();

            outQueue.push(task);
        }
    }

    void decoder(WorkQueue<ImageTask*>& inQueue, WorkQueue<ImageTask*>& outQueue) {
        int lastInitBatch = -1;

        while (true) {
            auto imgs = inQueue.popMax(static_cast<size_t>(m_batchSize));
            if (imgs.empty() && inQueue.isSentinel()) {
                break;
            }
            if (imgs.empty()) {
                continue;
            }

            std::vector<ImageTask*> validImgs;
            validImgs.reserve(imgs.size());

            std::vector<const unsigned char*> rawPointers;
            std::vector<size_t> rawLengths;
            std::vector<nvjpegImage_t> outImages;

            rawPointers.reserve(imgs.size());
            rawLengths.reserve(imgs.size());
            outImages.reserve(imgs.size());

            for (auto* img : imgs) {
                if (!img->h_data || img->h_size == 0) {
                    continue;
                }

                int nComponents = 0;
                nvjpegChromaSubsampling_t sampling;
                int widths[NVJPEG_MAX_COMPONENT];
                int heights[NVJPEG_MAX_COMPONENT];

                nvjpegStatus_t infoSt = nvjpegGetImageInfo(m_nvjHandle, img->h_data, img->h_size, &nComponents, &sampling, widths, heights);
                if (infoSt != NVJPEG_STATUS_SUCCESS) {
                    continue;
                }

                img->width = widths[0];
                img->height = heights[0];
                if (img->width <= 0 || img->height <= 0) {
                    continue;
                }

                size_t decodeBytes = static_cast<size_t>(img->width) * static_cast<size_t>(img->height) * 3;
                cudaCheck(cudaMalloc(&img->d_decoded, decodeBytes), "cudaMalloc(decoded) failed");

                nvjpegImage_t outImage;
                memset(&outImage, 0, sizeof(nvjpegImage_t));
                outImage.channel[0] = img->d_decoded;
                outImage.pitch[0] = static_cast<unsigned int>(img->width * 3);

                validImgs.push_back(img);
                rawPointers.push_back(img->h_data);
                rawLengths.push_back(img->h_size);
                outImages.push_back(outImage);
            }

            if (!validImgs.empty()) {
                if (lastInitBatch != static_cast<int>(validImgs.size())) {
                    nvjpegCheck(
                        nvjpegDecodeBatchedInitialize(m_nvjHandle, m_nvjDecoderState, static_cast<int>(validImgs.size()), m_readerThreads, NVJPEG_OUTPUT_RGBI),
                        "nvjpegDecodeBatchedInitialize failed"
                    );
                    lastInitBatch = static_cast<int>(validImgs.size());
                }

                nvjpegStatus_t batchSt = nvjpegDecodeBatched(m_nvjHandle, m_nvjDecoderState, rawPointers.data(), rawLengths.data(), outImages.data(), m_decodeStream);
                cudaCheck(cudaStreamSynchronize(m_decodeStream), "decode stream sync failed");

                if (batchSt != NVJPEG_STATUS_SUCCESS) {
                    for (auto* img : validImgs) {
                        if (!img->h_data || img->width <= 0 || img->height <= 0 || !img->d_decoded) {
                            continue;
                        }
                        nvjpegImage_t singleOut;
                        memset(&singleOut, 0, sizeof(nvjpegImage_t));
                        singleOut.channel[0] = img->d_decoded;
                        singleOut.pitch[0] = static_cast<unsigned int>(img->width * 3);
                        nvjpegStatus_t st = nvjpegDecode(m_nvjHandle, m_nvjSingleDecoderState, img->h_data, img->h_size, NVJPEG_OUTPUT_RGBI, &singleOut, m_decodeStream);
                        cudaCheck(cudaStreamSynchronize(m_decodeStream), "single decode stream sync failed");
                        if (st != NVJPEG_STATUS_SUCCESS) {
                            cudaFree(img->d_decoded);
                            img->d_decoded = nullptr;
                            img->width = 0;
                            img->height = 0;
                        }
                    }
                }
            }

            for (auto* img : imgs) {
                if (img->h_data) {
                    if (cudaFreeHost(img->h_data) != cudaSuccess) {
                        free(img->h_data);
                    }
                    img->h_data = nullptr;
                    img->h_size = 0;
                }
            }

            outQueue.pushMany(imgs);
        }
        outQueue.setSentinel();
    }

    void resizer(WorkQueue<ImageTask*>& inQueue, WorkQueue<ImageTask*>& outQueue) {
        dim3 block(32, 16, 1);

        while (true) {
            auto imgs = inQueue.popMax(static_cast<size_t>(m_batchSize));
            if (imgs.empty() && inQueue.isSentinel()) {
                break;
            }
            if (imgs.empty()) {
                continue;
            }

            std::vector<GpuData> gpuDataVec;
            gpuDataVec.reserve(imgs.size());
            std::vector<ImageTask*> okImgs;
            okImgs.reserve(imgs.size());

            for (auto* img : imgs) {
                if (!img->d_decoded || img->width <= 0 || img->height <= 0) {
                    continue;
                }
                cudaCheck(cudaMalloc(&img->d_resized, static_cast<size_t>(m_resize) * m_resize * sizeof(float)), "cudaMalloc(resized) failed");
                GpuData gd;
                gd.decodedPtr = img->d_decoded;
                gd.resizedPtr = img->d_resized;
                gd.originalWidth = img->width;
                gd.originalHeight = img->height;
                gpuDataVec.push_back(gd);
                okImgs.push_back(img);
            }

            if (!okImgs.empty()) {
                cudaCheck(cudaMemcpyAsync(m_d_gpuDataVec, gpuDataVec.data(), okImgs.size() * sizeof(GpuData), cudaMemcpyHostToDevice, m_resizeStream), "cudaMemcpyAsync(GpuData) failed");
                dim3 grid(
                    (m_resize + block.x - 1) / block.x,
                    (m_resize + block.y - 1) / block.y,
                    static_cast<unsigned int>(okImgs.size())
                );
                batchLanczosResizeKernel<<<grid, block, 0, m_resizeStream>>>(m_d_gpuDataVec, m_resize);
                cudaCheck(cudaGetLastError(), "resize kernel launch failed");
                cudaCheck(cudaStreamSynchronize(m_resizeStream), "resize stream sync failed");
            }

            for (auto* img : imgs) {
                if (img->d_decoded) {
                    cudaFree(img->d_decoded);
                    img->d_decoded = nullptr;
                }
            }

            outQueue.pushMany(imgs);
        }
        outQueue.setSentinel();
    }

    void hasher(WorkQueue<ImageTask*>& inQueue, uint8_t* d_hashBytes) {
        while (true) {
            auto imgs = inQueue.popMax(static_cast<size_t>(m_batchSize));
            if (imgs.empty() && inQueue.isSentinel()) {
                break;
            }
            if (imgs.empty()) {
                continue;
            }

            std::vector<ImageTask*> ok;
            ok.reserve(imgs.size());
            for (auto* img : imgs) {
                if (img->d_resized) {
                    ok.push_back(img);
                }
            }

            if (!ok.empty()) {
                int subB = static_cast<int>(ok.size());

                std::vector<const float*> h_A_array(subB, nullptr);
                std::vector<int> h_outPos(subB, 0);

                for (int i = 0; i < subB; ++i) {
                    h_A_array[i] = ok[i]->d_resized;
                    h_outPos[i] = ok[i]->index;
                }

                cudaCheck(cudaMemcpyAsync(m_d_AArray, h_A_array.data(), subB * sizeof(float*), cudaMemcpyHostToDevice, m_hashStream), "cudaMemcpyAsync(AArray) failed");
                cudaCheck(cudaMemcpyAsync(m_d_outPos, h_outPos.data(), subB * sizeof(int), cudaMemcpyHostToDevice, m_hashStream), "cudaMemcpyAsync(outPos) failed");

                phashDct8x8Kernel<<<subB, 64, 0, m_hashStream>>>(
                    m_d_AArray,
                    m_resize,
                    m_lowFreq,
                    m_d_outPos,
                    d_hashBytes,
                    subB
                );
                cudaCheck(cudaGetLastError(), "phash kernel launch failed");
                cudaCheck(cudaStreamSynchronize(m_hashStream), "hash stream sync failed");
            }

            for (auto* img : imgs) {
                if (img->d_resized) {
                    cudaFree(img->d_resized);
                    img->d_resized = nullptr;
                }
            }
        }
    }

    int m_resize;
    int m_lowFreq;
    int m_batchSize;
    int m_readerThreads;

    WorkQueue<ImageTask*> m_inputQueue;
    WorkQueue<ImageTask*> m_decodeQueue;
    WorkQueue<ImageTask*> m_resizeQueue;

    nvjpegHandle_t m_nvjHandle{ nullptr };
    nvjpegJpegState_t m_nvjDecoderState{ nullptr };
    nvjpegJpegState_t m_nvjSingleDecoderState{ nullptr };

    cudaStream_t m_decodeStream{ nullptr };
    cudaStream_t m_resizeStream{ nullptr };
    cudaStream_t m_hashStream{ nullptr };
    const float** m_d_AArray{ nullptr };
    int* m_d_outPos{ nullptr };
    GpuData* m_d_gpuDataVec{ nullptr };
};

} // namespace

namespace phash_cuda {

std::vector<std::array<uint8_t, 8>> compute_phash(
    const std::vector<std::string>& paths,
    int resize,
    int low_freq,
    int batch_size
) {
    if (paths.empty()) {
        return {};
    }

    int K = low_freq;
    if (K < 1) {
        K = 1;
    }
    if (K > 8) {
        K = 8;
    }
    if (resize > 0) {
        K = std::min(K, resize);
    }

    int threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads < 1) {
        threads = 4;
    }
    threads = std::min(32, threads);

    int B = batch_size;
    if (B < 1) {
        B = 1;
    }

    CudaPhashPipeline pipeline(resize, K, B, threads);
    return pipeline.computeBytes(paths);
}

}
