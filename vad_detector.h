#pragma once

#include <memory>
#include <string>
#include <vector>

// Silero VAD 的轻量封装。
//
// Feed() 可以连续喂任意长度的 16kHz float32 单声道音频。内部会自动切成
// Silero 所需的 512 采样点小块，并保留模型的循环状态。
class VadDetector {
 public:
  struct Result {
    bool has_speech = false;   // 本次 Feed 中是否出现人声
    bool speech_active = false;  // 流结束时是否仍处于人声段
    float max_probability = 0.0f;
  };

  explicit VadDetector(const std::string& model_path,
                       float threshold = 0.5f);
  ~VadDetector();

  VadDetector(const VadDetector&) = delete;
  VadDetector& operator=(const VadDetector&) = delete;

  // 清空模型状态，开始一条新的音频流。
  void Reset();

  // 连续流接口：不足 512 点的尾部会留到下次 Feed。
  Result Feed(const float* samples, size_t count);
  Result Feed(const std::vector<float>& samples) {
    return Feed(samples.data(), samples.size());
  }

  // 文件接口：从干净状态检测整段音频，并补零处理最后半块。
  Result Detect(const std::vector<float>& samples);

  int ChunkSizeSamples() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
