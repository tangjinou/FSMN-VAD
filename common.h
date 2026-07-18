#pragma once

#include <string>
#include <vector>

// =============================================================================
// 工程共享：常量、配置、检测结果、字符串小工具
// =============================================================================

// 与训练 / 导出 ONNX 保持一致
constexpr int kSampleRate = 16000;
constexpr int kMelBins = 80;
constexpr int kLfrM = 5;
constexpr int kLfrN = 3;
constexpr int kFeatureDim = kMelBins * kLfrM;  // 400
constexpr int kVocabSize = 2599;
constexpr int kBlankId = 0;

constexpr int kScoreBeamSize = 10;
constexpr int kPathBeamSize = 40;
constexpr float kMinTokenProb = 0.01f;

// 麦克风滑动窗
constexpr double kWindowSec = 2.0;
constexpr double kHopSec = 0.4;
constexpr double kCooldownSec = 1.5;
constexpr int kHopSamples = static_cast<int>(kSampleRate * kHopSec);

struct Options {
  std::string keywords = "大圣,悟空";
  std::string model_dir = "./onnx_model";
  std::string device;  // 空=默认麦；数字=编号；其它=名称子串
  std::string wav;     // 非空则离线读文件
  float min_score = 0.15f;
  float vad_threshold = 0.5f;
  int session_timeout_seconds = 10;
  bool verbose = false;
};

struct Detection {
  bool detected = false;
  std::string keyword;
  float score = 0.0f;
};

std::vector<std::string> SplitCsv(const std::string& text);
std::vector<std::string> Utf8Chars(const std::string& text);

// 「悟空」口语变体一并纳入；展示时仍显示「悟空」
std::vector<std::string> ExpandKeywords(const std::string& csv);
std::string DisplayName(const std::string& keyword);
