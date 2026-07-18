#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"

// KWS 引擎：PCM → 特征 → ONNX → CTC → Detection
// 实现细节（Fbank / CTC）都在 kws_engine.cc，读头文件只看「怎么用」。
class KwsEngine {
 public:
  explicit KwsEngine(const Options& options);
  ~KwsEngine();

  KwsEngine(const KwsEngine&) = delete;
  KwsEngine& operator=(const KwsEngine&) = delete;

  Detection Infer(const std::vector<float>& waveform);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
