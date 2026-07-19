#include "sound_tracking_action.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

namespace {

// ---------------------------------------------------------------------------
// 公共打印层：统一附带 remaining / timeout
// ---------------------------------------------------------------------------

std::string FormatSide(int direction) {
  if (direction < 0) return "（左侧）";
  if (direction > 0) return "（右侧）";
  return "（正前方）";
}

void PrintWithSessionClock(const std::string& tag, const std::string& message,
                           std::chrono::seconds remaining,
                           std::chrono::seconds timeout) {
  std::cout << tag << " " << message
            << " remaining=" << remaining.count() << "s"
            << " timeout=" << timeout.count() << "s\n";
}

void PrintSessionEvent(const std::string& event,
                       std::chrono::seconds remaining,
                       std::chrono::seconds timeout) {
  PrintWithSessionClock("[声源跟踪会话]", event, remaining, timeout);
}

void PrintLocator(int direction, std::chrono::seconds remaining,
                  std::chrono::seconds timeout) {
  PrintWithSessionClock(
      "[模拟声源定位]",
      "声音位于 " + std::to_string(direction) + "°" + FormatSide(direction),
      remaining, timeout);
}

void PrintServo(int from_angle, int to_angle, std::chrono::seconds remaining,
                std::chrono::seconds timeout) {
  PrintWithSessionClock("[模拟舵机]",
                        "从 " + std::to_string(from_angle) + "° 旋转到 " +
                            std::to_string(to_angle) + "°",
                        remaining, timeout);
}

}  // namespace

int SimulatedSoundLocator::DetectDirection() {
  // 相对正前方的角度：负数在左，正数在右。
  static constexpr std::array<int, 5> kDirections{-60, -30, 0, 30, 60};
  const int direction = kDirections[next_direction_ % kDirections.size()];
  ++next_direction_;
  return direction;
}

SimulatedServo::~SimulatedServo() { JoinWorker(); }

void SimulatedServo::JoinWorker() {
  if (worker_.joinable()) worker_.join();
}

int SimulatedServo::Angle() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return angle_degrees_;
}

bool SimulatedServo::IsRotating() const { return rotating_.load(); }

void SimulatedServo::RotateToDirectionAsync(int direction_degrees) {
  const int target_angle = std::clamp(direction_degrees + 90, 0, 180);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (target_angle == angle_degrees_ || rotating_.load()) return;
    rotating_.store(true);
  }

  JoinWorker();
  worker_ = std::thread([this, target_angle] {
    // 模拟舵机机械旋转耗时；在后台执行，避免堵住 VAD/会话超时计时。
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      angle_degrees_ = target_angle;
    }
    rotating_.store(false);
  });
}

SoundTrackingSession::SoundTrackingSession(std::chrono::seconds idle_timeout)
    : idle_timeout_(idle_timeout),
      expires_at_(std::chrono::steady_clock::now() + idle_timeout_) {}

bool SoundTrackingSession::IsExpired() const {
  return std::chrono::steady_clock::now() >= expires_at_;
}

std::chrono::seconds SoundTrackingSession::IdleTimeout() const {
  return idle_timeout_;
}

std::chrono::seconds SoundTrackingSession::Remaining() const {
  const auto now = std::chrono::steady_clock::now();
  if (now >= expires_at_) return std::chrono::seconds{0};
  // 向上取整到秒，避免刚创建时因截断显示成 timeout-1。
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(expires_at_ - now);
  return std::chrono::seconds{(ms.count() + 999) / 1000};
}

void SoundTrackingSession::OnVoice() {
  // 人声总是按墙钟刷新空闲超时，保证停说后约 timeout 秒结束。
  expires_at_ = std::chrono::steady_clock::now() + idle_timeout_;

  // 舵机旋转期间只续期，不叠加新的定位/旋转，避免主循环被拖慢。
  if (servo_.IsRotating()) return;

  const int direction = locator_.DetectDirection();
  const int from_angle = servo_.Angle();
  const int to_angle = std::clamp(direction + 90, 0, 180);
  PrintLocator(direction, Remaining(), idle_timeout_);
  PrintServo(from_angle, to_angle, Remaining(), idle_timeout_);
  servo_.RotateToDirectionAsync(direction);
}

SoundTrackingKeywordAction::SoundTrackingKeywordAction(
    std::chrono::seconds idle_timeout)
    : idle_timeout_(idle_timeout) {}

void SoundTrackingKeywordAction::OnKeyword(const KeywordHit& /*hit*/) {
  if (session_) {
    PrintSessionEvent("旧会话已删除", session_->Remaining(),
                      session_->IdleTimeout());
    session_.reset();
  }
  session_ = std::make_unique<SoundTrackingSession>(idle_timeout_);
  PrintSessionEvent("已创建", session_->Remaining(), session_->IdleTimeout());
}

void SoundTrackingKeywordAction::OnVoiceActivity(bool has_speech) {
  if (!session_) return;

  if (session_->IsExpired()) {
    PrintSessionEvent("已结束", std::chrono::seconds{0},
                      session_->IdleTimeout());
    session_.reset();
    return;
  }

  if (has_speech) session_->OnVoice();
}
