#pragma once

#include <chrono>
#include <memory>

#include "keyword_action.h"

// =============================================================================
// 模拟声源定位 + 舵机跟踪（KeywordAction 扩展）
// =============================================================================

// 模拟声源定位：每次检测依次返回左侧、正前方、右侧等方向。
// 方向角范围为 [-90, 90]，负数表示左侧，正数表示右侧。
class SimulatedSoundLocator {
 public:
  int DetectDirection();

 private:
  size_t next_direction_ = 0;
};

// 模拟舵机：把声源方向映射到 [0, 180] 度的舵机角度。
class SimulatedServo {
 public:
  int Angle() const { return angle_degrees_; }
  // 返回旋转后的舵机角度 [0, 180]。
  int RotateToDirection(int direction_degrees);

 private:
  int angle_degrees_ = 90;
};

// 关键词命中后创建的会话；每次人声都会刷新空闲超时并触发跟踪。
class SoundTrackingSession {
 public:
  explicit SoundTrackingSession(std::chrono::seconds idle_timeout);

  bool IsExpired() const;
  std::chrono::seconds IdleTimeout() const;
  std::chrono::seconds Remaining() const;
  void OnVoice();

 private:
  std::chrono::seconds idle_timeout_;
  std::chrono::steady_clock::time_point expires_at_;
  SimulatedSoundLocator locator_;
  SimulatedServo servo_;
};

// 管理声源跟踪会话：关键词创建会话，VAD 状态驱动会话行为。
class SoundTrackingKeywordAction : public KeywordAction {
 public:
  explicit SoundTrackingKeywordAction(std::chrono::seconds idle_timeout);

  void OnKeyword(const KeywordHit& hit) override;
  void OnVoiceActivity(bool has_speech) override;

 private:
  std::chrono::seconds idle_timeout_;
  std::unique_ptr<SoundTrackingSession> session_;
};
