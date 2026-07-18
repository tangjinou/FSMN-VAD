// main.cc — 程序入口（编排层）
//
// ============================================================================
// 先读这里看懂整条流水线，细节再下钻到各模块：
// ============================================================================
//
//   麦克风 mic_input.* ──┐
//   WAV 文件 ReadWav() ──┼──► VAD vad_detector.*
//                        │         │
//                        │    无人声 → 跳过
//                        │    有人声 ↓
//                        └──────────► KWS kws_engine.*
//                                         │
//                                    命中 → KeywordAction（可扩展）
//
// 模块职责：
//   common.*           共享常量 / Options / Detection / 字符串工具
//   mic_input.*        PortAudio 采集 + 滑动窗
//   vad_detector.*     Silero 人声门控
//   kws_engine.*       Fbank + ONNX + CTC 关键词识别
//   keyword_action.*   命中后的动作（打印 / 未来 GPIO·上报 等）
//   main.cc            命令行、WAV 读取、把上面模块串起来
//
// onnx_model/：
//   silero_vad.onnx / fsmn_kws.onnx / am.mvn.dim80_l2r2
//   tokens_2599.txt / lexicon.txt

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "keyword_action.h"
#include "kws_engine.h"
#include "mic_input.h"
#include "vad_detector.h"

namespace {

// =============================================================================
// 命令行
// =============================================================================

void PrintUsage(const char* prog) {
  std::cout << "用法: " << prog
            << " [--keyword 大圣,悟空] [--model-dir DIR] [--device DEVICE]\n"
            << "       [--min-score 0.15] [--vad-threshold 0.5]\n"
            << "       [--verbose] [--wav FILE.wav]\n";
}

bool ParseArgs(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (++i >= argc) {
        std::cerr << "缺少参数: " << name << "\n";
        return {};
      }
      return argv[i];
    };

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--keyword") {
      opt->keywords = need("--keyword");
    } else if (arg == "--model-dir") {
      opt->model_dir = need("--model-dir");
    } else if (arg == "--device") {
      opt->device = need("--device");
    } else if (arg == "--min-score") {
      opt->min_score = std::stof(need("--min-score"));
    } else if (arg == "--vad-threshold") {
      opt->vad_threshold = std::stof(need("--vad-threshold"));
    } else if (arg == "--verbose") {
      opt->verbose = true;
    } else if (arg == "--wav") {
      opt->wav = need("--wav");
    } else {
      std::cerr << "未知参数: " << arg << "\n";
      return false;
    }
  }
  return !opt->keywords.empty() && !opt->model_dir.empty() &&
         opt->vad_threshold > 0.0f && opt->vad_threshold < 1.0f;
}

// =============================================================================
// 简易 WAV：16kHz / 单声道 / PCM16
// =============================================================================

std::vector<float> ReadWav(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("无法打开 WAV");

  std::array<char, 44> header{};
  in.read(header.data(), header.size());
  if (!in || std::memcmp(header.data(), "RIFF", 4) != 0 ||
      std::memcmp(header.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("仅支持标准 PCM WAV");
  }

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t sample_rate = 0;
  std::memcpy(&format, header.data() + 20, 2);
  std::memcpy(&channels, header.data() + 22, 2);
  std::memcpy(&sample_rate, header.data() + 24, 4);
  std::memcpy(&bits, header.data() + 34, 2);
  if (format != 1 || channels != 1 || sample_rate != kSampleRate ||
      bits != 16) {
    throw std::runtime_error("WAV 必须是 16kHz/单声道/PCM16");
  }

  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  std::vector<float> samples(bytes.size() / 2);
  for (size_t i = 0; i < samples.size(); ++i) {
    int16_t value = 0;
    std::memcpy(&value, bytes.data() + i * 2, 2);
    samples[i] = value / 32768.0f;
  }
  return samples;
}

KeywordHit MakeHit(const Detection& result, bool from_microphone) {
  return KeywordHit{result.keyword, result.score, from_microphone};
}

// =============================================================================
// 运行模式：先 VAD，有人声才跑 KWS；命中后交给 KeywordAction
// =============================================================================

int RunWavFile(VadDetector* vad, KwsEngine* engine, KeywordAction* action,
               const Options& options) {
  const std::vector<float> waveform = ReadWav(options.wav);
  const VadDetector::Result voice = vad->Detect(waveform);
  if (options.verbose) {
    std::cout << "[vad] speech=" << voice.has_speech
              << " max_probability=" << voice.max_probability << "\n";
  }
  if (!voice.has_speech) {
    std::cout << "未检测到人声，跳过关键词识别\n";
    return 2;
  }

  const Detection result = engine->Infer(waveform);
  if (!result.detected) {
    std::cout << "未识别\n";
    return 2;
  }

  action->OnKeyword(MakeHit(result, /*from_microphone=*/false));
  // 离线自检额外打印一行，方便 make/cmake test 对照。
  std::cout << "keyword=" << result.keyword << " score=" << result.score
            << "\n";
  return 0;
}

int RunMicrophone(VadDetector* vad, KwsEngine* engine, KeywordAction* action,
                  const Options& options) {
  MicInput::InstallSignalHandlers();

  MicInput mic(kSampleRate, kHopSec, kWindowSec, options.device);
  mic.Start();
  std::cout << "输入设备: [" << mic.DeviceIndex() << "] " << mic.DeviceName()
            << "\n开始监听，请说“";

  const auto words = SplitCsv(options.keywords);
  for (size_t i = 0; i < words.size(); ++i) {
    if (i) std::cout << "”或“";
    std::cout << words[i];
  }
  std::cout << "”（Ctrl+C 退出）\n";

  // 冷却：相邻滑动窗不要连触发动作
  auto last_hit =
      std::chrono::steady_clock::now() - std::chrono::milliseconds(2000);

  bool first_window = true;
  std::vector<float> window;
  while (mic.ReadWindow(&window)) {
    // 首窗喂完整 2 秒给 VAD；之后只喂新增的 0.4 秒，避免重叠重复计分
    const float* new_audio = window.data();
    size_t new_audio_size = window.size();
    if (!first_window && window.size() > static_cast<size_t>(kHopSamples)) {
      new_audio = window.data() + window.size() - kHopSamples;
      new_audio_size = kHopSamples;
    }
    first_window = false;

    const VadDetector::Result voice = vad->Feed(new_audio, new_audio_size);
    if (options.verbose) {
      std::cout << "[vad] speech=" << voice.has_speech
                << " active=" << voice.speech_active
                << " max_probability=" << voice.max_probability << "\n";
    }

    // 门控：静音不跑 Fbank / FSMN-KWS
    if (!voice.has_speech && !voice.speech_active) continue;

    const Detection result = engine->Infer(window);
    const auto now = std::chrono::steady_clock::now();
    const double gap =
        std::chrono::duration<double>(now - last_hit).count();
    if (!result.detected || gap < kCooldownSec) continue;

    action->OnKeyword(MakeHit(result, /*from_microphone=*/true));
    last_hit = now;
  }

  mic.Stop();
  std::cout << "已停止监听\n";
  return 0;
}

}  // namespace

// =============================================================================
// 入口
// =============================================================================

int main(int argc, char** argv) {
  try {
    std::cout << std::unitbuf;

    Options options;
    if (!ParseArgs(argc, argv, &options)) return 1;

    std::cout << "加载 ONNX 模型: " << options.model_dir << "\n";
    const auto model_dir = std::filesystem::path(options.model_dir);

    VadDetector vad((model_dir / "silero_vad.onnx").string(),
                    options.vad_threshold);
    KwsEngine engine(options);
    auto action = CreateDefaultKeywordAction(options);

    if (!options.wav.empty()) {
      return RunWavFile(&vad, &engine, action.get(), options);
    }
    return RunMicrophone(&vad, &engine, action.get(), options);
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << "\n";
    return 1;
  }
}
