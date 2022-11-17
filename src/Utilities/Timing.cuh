#ifndef GPUEM_TIMING_CUH
#define GPUEM_TIMING_CUH

/******** Cuda Event Timer ********/
struct cudaTimer {
    float time = 0.0;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    cudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~cudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() const {
        cudaEventRecord(start_event);
    }

    void stop() const {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    float elapsed() {
        cudaEventElapsedTime(&time, start_event, stop_event);
        return time;
    }

    float get_time() const {
        return time / 1000.0f;
    }
};

/******** CPU Timer ********/
struct cpuTimer {
    double time = 0.0;
    std::chrono::steady_clock::time_point start_event;
    std::chrono::steady_clock::time_point stop_event;

    void start() {
        start_event = std::chrono::steady_clock::now();
    }

    void stop() {
        stop_event = std::chrono::steady_clock::now();
    }

    float elapsed() {
        std::chrono::duration<double, std::milli> dur = (stop_event - start_event);
        time = dur.count();
        return time;
    }

    double get_time() const {
        return time / 1000.0;
    }
};

#endif //GPUEM_TIMING_CUH
