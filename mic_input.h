#pragma once

#include <string>
#include <vector>

// 麦克风输入：PortAudio 采集 float32 单声道，并按 hop/window 输出滑动窗。
class MicInput {
 public:
  MicInput(int sample_rate, double hop_seconds, double window_seconds,
           const std::string& device_spec = "");
  ~MicInput();

  MicInput(const MicInput&) = delete;
  MicInput& operator=(const MicInput&) = delete;

  void Start();
  void Stop();

  // 读满一个滑动窗返回 true；停止信号或关闭后返回 false。
  bool ReadWindow(std::vector<float>* window);

  int DeviceIndex() const { return device_index_; }
  const std::string& DeviceName() const { return device_name_; }

  static void RequestStop();
  static void InstallSignalHandlers();

 private:
  struct Impl;
  Impl* impl_;

  int sample_rate_;
  int hop_samples_;
  int window_samples_;
  std::string device_spec_;
  int device_index_ = -1;
  std::string device_name_;
};
