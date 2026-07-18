// vad_detector.cc
//
// Silero VAD ONNX 流式推理。
// 模型每次接收 512 个新采样点，并额外携带：
//   - 64 个上一块的上下文采样
//   - [2, 1, 128] 的循环状态
// 两者都必须传给下一次推理，否则连续语音的判断会不稳定。

#include "vad_detector.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 512;    // 32 ms
constexpr int kContextSamples = 64;   // 4 ms
constexpr int kInputSamples = kChunkSamples + kContextSamples;
constexpr int kStateSize = 2 * 1 * 128;
constexpr int kMinSilenceSamples = kSampleRate * 100 / 1000;  // 100 ms
constexpr float kEndThresholdOffset = 0.15f;

}  // namespace

struct VadDetector::Impl {
  explicit Impl(const std::string& model_path, float speech_threshold)
      : threshold(speech_threshold),
        env(ORT_LOGGING_LEVEL_ERROR, "silero-vad"),
        session(nullptr),
        memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                          OrtMemTypeDefault)) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, model_path.c_str(), session_options);
    Reset();
  }

  void Reset() {
    std::fill(context.begin(), context.end(), 0.0f);
    std::fill(state.begin(), state.end(), 0.0f);
    pending.clear();
    speech_active = false;
    silence_samples = 0;
  }

  // 运行一个 32ms 小块，并把 stateN/context 留给下一块。
  float PredictChunk(const float* chunk) {
    std::copy(context.begin(), context.end(), input.begin());
    std::copy(chunk, chunk + kChunkSamples,
              input.begin() + kContextSamples);

    const std::array<int64_t, 2> input_shape{1, kInputSamples};
    const std::array<int64_t, 3> state_shape{2, 1, 128};
    const std::array<int64_t, 1> sample_rate_shape{1};

    std::array<Ort::Value, 3> inputs{
        Ort::Value::CreateTensor<float>(
            memory, input.data(), input.size(), input_shape.data(),
            input_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory, state.data(), state.size(), state_shape.data(),
            state_shape.size()),
        Ort::Value::CreateTensor<int64_t>(
            memory, sample_rate.data(), sample_rate.size(),
            sample_rate_shape.data(), sample_rate_shape.size())};

    static constexpr const char* kInputNames[] = {"input", "state", "sr"};
    static constexpr const char* kOutputNames[] = {"output", "stateN"};
    auto outputs =
        session.Run(Ort::RunOptions{nullptr}, kInputNames, inputs.data(),
                    inputs.size(), kOutputNames, 2);

    const float probability = outputs[0].GetTensorData<float>()[0];
    const float* next_state = outputs[1].GetTensorData<float>();
    std::copy(next_state, next_state + kStateSize, state.begin());
    std::copy(input.end() - kContextSamples, input.end(), context.begin());
    return probability;
  }

  // 与参考 vad_detect 一致：进入人声后，需连续约 100ms 低概率才结束。
  // 双阈值可避免概率在边界附近反复切换。
  bool UpdateSpeechState(float probability) {
    if (probability >= threshold) {
      speech_active = true;
      silence_samples = 0;
    } else if (speech_active &&
               probability < threshold - kEndThresholdOffset) {
      silence_samples += kChunkSamples;
      if (silence_samples >= kMinSilenceSamples) {
        speech_active = false;
        silence_samples = 0;
      }
    }
    return probability >= threshold;
  }

  float threshold;
  bool speech_active = false;
  int silence_samples = 0;

  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  Ort::MemoryInfo memory;

  std::array<float, kContextSamples> context{};
  std::array<float, kStateSize> state{};
  std::array<float, kInputSamples> input{};
  std::array<int64_t, 1> sample_rate{kSampleRate};
  std::vector<float> pending;
};

VadDetector::VadDetector(const std::string& model_path, float threshold)
    : impl_([&] {
        if (threshold <= 0.0f || threshold >= 1.0f) {
          throw std::invalid_argument("VAD threshold 必须在 0 和 1 之间");
        }
        return std::make_unique<Impl>(model_path, threshold);
      }()) {}

VadDetector::~VadDetector() = default;

void VadDetector::Reset() { impl_->Reset(); }

VadDetector::Result VadDetector::Feed(const float* samples, size_t count) {
  Result result;
  impl_->pending.insert(impl_->pending.end(), samples, samples + count);

  size_t consumed = 0;
  while (impl_->pending.size() - consumed >= kChunkSamples) {
    const float probability =
        impl_->PredictChunk(impl_->pending.data() + consumed);
    result.max_probability = std::max(result.max_probability, probability);
    result.has_speech |= impl_->UpdateSpeechState(probability);
    consumed += kChunkSamples;
  }

  if (consumed > 0) {
    impl_->pending.erase(impl_->pending.begin(),
                         impl_->pending.begin() + consumed);
  }
  result.speech_active = impl_->speech_active;
  return result;
}

VadDetector::Result VadDetector::Detect(const std::vector<float>& samples) {
  Reset();
  Result result = Feed(samples);

  // 文件末尾不足 32ms 时补零，使最后一点音频也参与判断。
  if (!impl_->pending.empty()) {
    std::array<float, kChunkSamples> padded{};
    std::copy(impl_->pending.begin(), impl_->pending.end(), padded.begin());
    const float probability = impl_->PredictChunk(padded.data());
    result.max_probability = std::max(result.max_probability, probability);
    result.has_speech |= impl_->UpdateSpeechState(probability);
    impl_->pending.clear();
  }
  result.speech_active = impl_->speech_active;
  return result;
}

int VadDetector::ChunkSizeSamples() const { return kChunkSamples; }
