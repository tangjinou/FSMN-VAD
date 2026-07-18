#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"

// =============================================================================
// 关键词命中后的动作（可扩展模块）
//
// 检测流水线（VAD/KWS）只负责「有没有命中」；
// 「命中之后做什么」全部走 KeywordAction，便于日后换成：
//   - 串口 / GPIO 唤醒
//   - HTTP / MQTT 上报
//   - 本地日志落盘
//   - 真实麦克风阵列定位 / 舵机控制
// 而不改 main / kws_engine。
// =============================================================================

struct KeywordHit {
  std::string keyword;
  float score = 0.0f;
  bool from_microphone = false;  // true=实时麦；false=离线 WAV
};

// 动作接口：关键词命中和 VAD 人声状态都可分发给动作。
class KeywordAction {
 public:
  virtual ~KeywordAction() = default;
  virtual void OnKeyword(const KeywordHit& hit) = 0;
  virtual void OnVoiceActivity(bool) {}
};

// 默认动作：打印「已识别」（当前 demo 行为）。
class ConsoleKeywordAction : public KeywordAction {
 public:
  explicit ConsoleKeywordAction(bool verbose = false);
  void OnKeyword(const KeywordHit& hit) override;

 private:
  bool verbose_;
};

// 组合多个动作：一次命中依次调用（方便以后叠多个模块）。
class KeywordActionList : public KeywordAction {
 public:
  void Add(std::unique_ptr<KeywordAction> action);
  void OnKeyword(const KeywordHit& hit) override;
  void OnVoiceActivity(bool has_speech) override;

 private:
  std::vector<std::unique_ptr<KeywordAction>> actions_;
};

// 工厂：按 Options 组装当前需要的动作链。
std::unique_ptr<KeywordAction> CreateDefaultKeywordAction(const Options& options);
