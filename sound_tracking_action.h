#pragma once

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
  void RotateToDirection(int direction_degrees);

 private:
  int angle_degrees_ = 90;
};

// 关键词命中后模拟声源定位，并驱动模拟舵机朝向声源。
class SoundTrackingKeywordAction : public KeywordAction {
 public:
  void OnKeyword(const KeywordHit& hit) override;

 private:
  SimulatedSoundLocator locator_;
  SimulatedServo servo_;
};
