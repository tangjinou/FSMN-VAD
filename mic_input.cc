// mic_input.cc
//
// 麦克风采集：PortAudio 读入 16kHz float32 单声道，
// 再按 hop / window 切成滑动窗，供上层做关键词检测。

#include "mic_input.h"

#include <portaudio.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// 运行状态（Ctrl+C / SIGTERM 会置为 false）
// ---------------------------------------------------------------------------

std::atomic<bool> g_running{true};

void OnSignal(int) { g_running.store(false); }

// ---------------------------------------------------------------------------
// 线程安全音频队列：回调线程 Push，主线程 Pop
// ---------------------------------------------------------------------------

class AudioQueue {
 public:
  void Push(std::vector<float> chunk) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      chunks_.push_back(std::move(chunk));
    }
    condition_.notify_one();
  }

  // 有数据返回 true；超时或已停止且队列空则返回 false。
  bool Pop(std::vector<float>* chunk) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait_for(lock, std::chrono::milliseconds(100), [&] {
      return !chunks_.empty() || !g_running.load();
    });
    if (chunks_.empty()) return false;
    *chunk = std::move(chunks_.front());
    chunks_.pop_front();
    return true;
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::vector<float>> chunks_;
};

// PortAudio 回调：把本帧采样拷进队列。
int AudioCallback(const void* input, void* /*output*/, unsigned long frames,
                  const PaStreamCallbackTimeInfo* /*time_info*/,
                  PaStreamCallbackFlags /*flags*/, void* user_data) {
  if (input == nullptr) return paContinue;

  const float* samples = static_cast<const float*>(input);
  static_cast<AudioQueue*>(user_data)->Push(
      std::vector<float>(samples, samples + frames));
  return paContinue;
}

// ---------------------------------------------------------------------------
// 设备解析
// ---------------------------------------------------------------------------

// 空串 → 默认设备；纯数字 → 设备编号；其它 → 名称子串匹配。
int ResolveInputDevice(const std::string& spec) {
  if (spec.empty()) return Pa_GetDefaultInputDevice();

  const bool numeric = std::all_of(
      spec.begin(), spec.end(),
      [](unsigned char c) { return std::isdigit(c) != 0; });
  if (numeric) return std::stoi(spec);

  for (int i = 0; i < Pa_GetDeviceCount(); ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (info == nullptr || info->maxInputChannels <= 0) continue;
    if (std::strstr(info->name, spec.c_str()) != nullptr) return i;
  }
  return paNoDevice;
}

const PaDeviceInfo* RequireInputDevice(int device_index) {
  const PaDeviceInfo* info =
      device_index == paNoDevice ? nullptr : Pa_GetDeviceInfo(device_index);
  if (info == nullptr || info->maxInputChannels <= 0) {
    throw std::runtime_error("找不到输入设备");
  }
  return info;
}

// ---------------------------------------------------------------------------
// PortAudio 辅助
// ---------------------------------------------------------------------------

void CheckPa(PaError error, const char* what) {
  if (error != paNoError) {
    throw std::runtime_error(std::string(what) + ": " + Pa_GetErrorText(error));
  }
}

PaStreamParameters MakeInputParams(int device_index,
                                   const PaDeviceInfo* info) {
  PaStreamParameters params{};
  params.device = device_index;
  params.channelCount = 1;
  params.sampleFormat = paFloat32;
  params.suggestedLatency = info->defaultLowInputLatency;
  return params;
}

// 从累计 buffer 取出一个 window，并按 hop 向前滑动。
bool TryTakeWindow(std::vector<float>* buffer, int window_samples,
                   int hop_samples, std::vector<float>* window) {
  if (buffer->size() < static_cast<size_t>(window_samples)) return false;

  window->assign(buffer->end() - window_samples, buffer->end());
  // 丢掉 hop，保留 window - hop，形成下一次的重叠部分。
  buffer->assign(buffer->end() - (window_samples - hop_samples), buffer->end());
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// MicInput 实现细节
// ---------------------------------------------------------------------------

struct MicInput::Impl {
  AudioQueue queue;
  PaStream* stream = nullptr;
  bool portaudio_ready = false;
  std::vector<float> buffer;  // 累计原始采样，供滑动窗裁剪
};

MicInput::MicInput(int sample_rate, double hop_seconds, double window_seconds,
                   const std::string& device_spec)
    : impl_(new Impl),
      sample_rate_(sample_rate),
      hop_samples_(static_cast<int>(sample_rate * hop_seconds)),
      window_samples_(static_cast<int>(sample_rate * window_seconds)),
      device_spec_(device_spec) {}

MicInput::~MicInput() {
  Stop();
  delete impl_;
  impl_ = nullptr;
}

void MicInput::RequestStop() { g_running.store(false); }

void MicInput::InstallSignalHandlers() {
  signal(SIGINT, OnSignal);
  signal(SIGTERM, OnSignal);
}

void MicInput::Start() {
  if (impl_->stream != nullptr) return;

  g_running.store(true);

  CheckPa(Pa_Initialize(), "Pa_Initialize");
  impl_->portaudio_ready = true;

  try {
    device_index_ = ResolveInputDevice(device_spec_);
    const PaDeviceInfo* info = RequireInputDevice(device_index_);
    device_name_ = info->name;

    PaStreamParameters params = MakeInputParams(device_index_, info);
    CheckPa(Pa_OpenStream(&impl_->stream, &params, nullptr, sample_rate_,
                          hop_samples_, paClipOff, AudioCallback,
                          &impl_->queue),
            "Pa_OpenStream");
    CheckPa(Pa_StartStream(impl_->stream), "Pa_StartStream");
  } catch (...) {
    Stop();
    throw;
  }
}

void MicInput::Stop() {
  g_running.store(false);

  if (impl_->stream != nullptr) {
    Pa_StopStream(impl_->stream);
    Pa_CloseStream(impl_->stream);
    impl_->stream = nullptr;
  }
  if (impl_->portaudio_ready) {
    Pa_Terminate();
    impl_->portaudio_ready = false;
  }
}

bool MicInput::ReadWindow(std::vector<float>* window) {
  while (g_running.load()) {
    std::vector<float> chunk;
    if (!impl_->queue.Pop(&chunk)) continue;

    impl_->buffer.insert(impl_->buffer.end(), chunk.begin(), chunk.end());
    if (TryTakeWindow(&impl_->buffer, window_samples_, hop_samples_, window)) {
      return true;
    }
  }
  return false;
}
