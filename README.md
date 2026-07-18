# FSMN-KWS ONNXRuntime C++ Demo
本模块的目的是可以在1126B等本地硬件上能够接收关键字的唤起工作
关键字 如 大圣/悟空 等

为了debug更加低，所以工程需要支持mac环境和arm板子环境


本目录是可独立编译和运行的 macOS x86_64/Rosetta 版本，不依赖 FunASR、
Silero VAD 源码目录、Python、Homebrew 或系统安装的 ONNXRuntime/PortAudio。

## 目录（建议按此顺序阅读）

```text
音频 ──► VAD ──►(有人声)──► KWS ──► 「已识别」
```

- `main.cc`：入口编排（命令行 / WAV / 麦克风循环）
- `common.h` / `common.cc`：共享常量、Options、Detection、字符串工具
- `mic_input.h` / `mic_input.cc`：麦克风采集与滑动窗
- `vad_detector.h` / `vad_detector.cc`：Silero VAD 人声门控
- `kws_engine.h` / `kws_engine.cc`：Fbank + FSMN ONNX + CTC 关键词识别
- `keyword_action.h` / `keyword_action.cc`：命中后的动作（默认可打印；可扩展）
- `onnx_model/`：FSMN-KWS、Silero VAD、CMVN、token、词典
- `third_party/kaldi-native-fbank/`：Fbank 特征源码
- `include/` / `lib/`：ONNXRuntime、PortAudio（本地闭环）
- `test/`：大圣正样本

## 编译

需要 macOS Command Line Tools 和 CMake：

```bash
cd /Users/tjo/FunASR/FSMN-VAD
cmake -S . -B build
cmake --build build -j          # 生成 ./mic_demo
cmake --build build --target run       # 等价 make run
cmake --build build --target run_test  # 等价 make test
```

在源码根目录生成 `mic_demo`（与相邻的 `lib/`、`onnx_model/` 就地可用）。

## 麦克风运行

```bash
./mic_demo --verbose
```

默认识别“大圣”和“悟空”，并支持 `wǔ kōng` 发音变体。
程序先运行 Silero VAD；只有检测到人声时才计算 Fbank 和 FSMN-KWS，
静音窗口会直接跳过。

自定义关键词：

```bash
./mic_demo --keyword "大圣,悟空,八戒" --verbose
```

调整判定阈值：

```bash
./mic_demo --min-score 0.08 --verbose
```

调整 VAD 人声阈值（默认 `0.5`，越低越敏感）：

```bash
./mic_demo --vad-threshold 0.4 --verbose
```

## 离线自检

```bash
cmake --build build --target run_test
```

预期输出包含：

```text
已识别
keyword=大圣
```

## 说明

打包的二进制依赖为 x86_64；Apple Silicon Mac 会通过 Rosetta 运行。若需要
原生 arm64 版本，需要换成 arm64 ONNXRuntime 和 PortAudio 后重新编译。
