#include "sound_tracking_action.h"

#include <algorithm>
#include <array>
#include <iostream>

int SimulatedSoundLocator::DetectDirection() {
  // 相对正前方的角度：负数在左，正数在右。
  static constexpr std::array<int, 5> kDirections{-60, -30, 0, 30, 60};
  const int direction = kDirections[next_direction_ % kDirections.size()];
  ++next_direction_;
  std::cout << "[模拟声源定位] 声音位于 " << direction << "°"
            << (direction < 0 ? "（左侧）"
                              : direction > 0 ? "（右侧）" : "（正前方）")
            << "\n";
  return direction;
}

void SimulatedServo::RotateToDirection(int direction_degrees) {
  const int target_angle = std::clamp(direction_degrees + 90, 0, 180);
  std::cout << "[模拟舵机] 从 " << angle_degrees_ << "° 旋转到 "
            << target_angle << "°\n";
  angle_degrees_ = target_angle;
}

void SoundTrackingKeywordAction::OnKeyword(const KeywordHit& /*hit*/) {
  servo_.RotateToDirection(locator_.DetectDirection());
}
