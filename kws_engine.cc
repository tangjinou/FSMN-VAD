// kws_engine.cc
//
// 关键词识别核心，读顺序：
//   1. FeatureExtractor  — PCM → [T, 400]
//   2. KeywordDecoder    — logits → CTC beam → 关键词命中
//   3. KwsEngine::Impl   — 把上面两步串起来

#include "kws_engine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace {

// =============================================================================
// 1. 特征前端：PCM → Fbank(80) → LFR(5/3) → CMVN → [T, 400]
// =============================================================================

class FeatureExtractor {
 public:
  bool LoadCmvn(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("<AddShift>", 0) == 0) {
        if (!std::getline(in, line)) return false;
        means_ = ParseKaldiVector(line);
      } else if (line.rfind("<Rescale>", 0) == 0) {
        if (!std::getline(in, line)) return false;
        scales_ = ParseKaldiVector(line);
      }
    }
    return means_.size() == static_cast<size_t>(kFeatureDim) &&
           scales_.size() == static_cast<size_t>(kFeatureDim);
  }

  // feats[t * 400 + d]；*num_frames = T
  std::vector<float> Compute(const std::vector<float>& waveform,
                             int* num_frames) const {
    knf::OnlineFbank fbank(MakeFbankOptions());

    // fbank 按接近 int16 的幅度算，float PCM 先 ×32768
    std::vector<float> scaled(waveform.size());
    std::transform(waveform.begin(), waveform.end(), scaled.begin(),
                   [](float x) { return x * 32768.0f; });
    fbank.AcceptWaveform(kSampleRate, scaled.data(),
                         static_cast<int32_t>(scaled.size()));

    const int n = fbank.NumFramesReady();
    if (n <= 0) {
      *num_frames = 0;
      return {};
    }

    std::vector<std::vector<float>> fbank_frames(n,
                                                 std::vector<float>(kMelBins));
    for (int i = 0; i < n; ++i) {
      const float* row = fbank.GetFrame(i);
      std::copy(row, row + kMelBins, fbank_frames[i].begin());
    }
    return ApplyLfrAndCmvn(fbank_frames, num_frames);
  }

 private:
  static knf::FbankOptions MakeFbankOptions() {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = kSampleRate;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.dither = 1.0f;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = kMelBins;
    opts.use_energy = false;
    return opts;
  }

  // Kaldi 行：<LearnRateCoef> 0 [ v0 v1 ... ]
  static std::vector<float> ParseKaldiVector(const std::string& line) {
    std::istringstream in(line);
    std::string token;
    in >> token >> token >> token;
    std::vector<float> values;
    while (in >> token && token != "]") values.push_back(std::stof(token));
    return values;
  }

  std::vector<float> ApplyLfrAndCmvn(
      const std::vector<std::vector<float>>& fbank_frames,
      int* num_frames) const {
    const int n_fbank = static_cast<int>(fbank_frames.size());
    const int n_lfr = (n_fbank + kLfrN - 1) / kLfrN;
    const int left_pad = (kLfrM - 1) / 2;

    std::vector<float> out(static_cast<size_t>(n_lfr) * kFeatureDim);
    for (int t = 0; t < n_lfr; ++t) {
      for (int m = 0; m < kLfrM; ++m) {
        const int src =
            std::clamp(t * kLfrN + m - left_pad, 0, n_fbank - 1);
        for (int d = 0; d < kMelBins; ++d) {
          const int dim = m * kMelBins + d;
          out[static_cast<size_t>(t) * kFeatureDim + dim] =
              (fbank_frames[src][d] + means_[dim]) * scales_[dim];
        }
      }
    }
    *num_frames = n_lfr;
    return out;
  }

  std::vector<float> means_;
  std::vector<float> scales_;
};

// =============================================================================
// 2. 关键词解码：logits → CTC beam → 子串匹配
//
// CTC 直觉（blank = 0）：
//   _ 大 大 _ 圣 _  → 「大圣」
//
// 每条路径拆两路概率：blank 结尾 / 非 blank 结尾，
// 才能正确处理「同一字重复」与「blank 后再发一次」。
// =============================================================================

class KeywordDecoder {
 public:
  bool Load(const std::filesystem::path& model_dir,
            const std::string& keywords_csv) {
    return LoadTokens(model_dir / "tokens_2599.txt") &&
           LoadLexicon(model_dir / "lexicon.txt", keywords_csv) &&
           BuildKeywordTokenIds(keywords_csv);
  }

  Detection Decode(const float* logits, int num_frames, int vocab_size) const {
    return MatchKeywords(RunCtcBeam(logits, num_frames, vocab_size));
  }

 private:
  struct EmittedToken {
    int token_id = 0;
    int best_frame = 0;
    float best_prob = 0.0f;
  };

  struct PathState {
    double blank_ending = 0.0;
    double non_blank_ending = 0.0;
    std::vector<EmittedToken> emitted;
    double Total() const { return blank_ending + non_blank_ending; }
  };

  using Beam = std::map<std::vector<int>, PathState>;

  bool Fail(const std::string& message) const {
    std::cerr << message << "\n";
    return false;
  }

  bool LoadTokens(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) return Fail("无法打开 tokens_2599.txt");

    std::string line;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      tokens_.push_back(line);
    }
    if (tokens_.size() != static_cast<size_t>(kVocabSize)) {
      return Fail("token 数量不是 2599");
    }
    for (size_t i = 0; i < tokens_.size(); ++i) {
      token_to_id_[tokens_[i]] = static_cast<int>(i);
    }
    return true;
  }

  // lexicon 很大：只加载当前关键词及相关单字
  bool LoadLexicon(const std::filesystem::path& path,
                   const std::string& keywords_csv) {
    const auto expanded = ExpandKeywords(keywords_csv);
    std::set<std::string> wanted(expanded.begin(), expanded.end());
    for (const auto& keyword : expanded) {
      for (const auto& ch : Utf8Chars(keyword)) wanted.insert(ch);
    }

    std::ifstream file(path);
    if (!file) return Fail("无法打开 lexicon.txt");

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream in(line);
      std::string word;
      in >> word;
      if (!wanted.count(word)) continue;
      lexicon_[word] = {std::istream_iterator<std::string>{in}, {}};
    }
    return true;
  }

  // 「大圣」→ 大 圣；「悟空」→ lexicon「物 空」
  std::vector<std::string> ResolveAcousticPieces(
      const std::string& keyword) const {
    if (const auto it = lexicon_.find(keyword); it != lexicon_.end()) {
      return it->second;
    }

    std::vector<std::string> pieces;
    for (const auto& ch : Utf8Chars(keyword)) {
      if (token_to_id_.count(ch)) {
        pieces.push_back(ch);
      } else if (const auto lit = lexicon_.find(ch); lit != lexicon_.end()) {
        pieces.insert(pieces.end(), lit->second.begin(), lit->second.end());
      } else {
        pieces.push_back(ch);
      }
    }
    return pieces;
  }

  bool BuildKeywordTokenIds(const std::string& keywords_csv) {
    keyword_token_ids_.clear();
    searchable_token_ids_ = {kBlankId};

    for (const auto& keyword : ExpandKeywords(keywords_csv)) {
      std::vector<int> ids;
      for (const auto& piece : ResolveAcousticPieces(keyword)) {
        const auto it = token_to_id_.find(piece);
        if (it == token_to_id_.end()) {
          std::cerr << "关键词 token 不存在: " << keyword << " -> " << piece
                    << "\n";
          ids.clear();
          break;
        }
        ids.push_back(it->second);
        searchable_token_ids_.insert(it->second);
      }
      if (!ids.empty()) keyword_token_ids_[keyword] = std::move(ids);
    }
    return !keyword_token_ids_.empty();
  }

  std::vector<std::pair<int, float>> CollectFrameCandidates(
      const float* logits_row, int vocab_size) const {
    const float max_logit =
        *std::max_element(logits_row, logits_row + vocab_size);

    double denom = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
      denom += std::exp(logits_row[i] - max_logit);
    }

    std::vector<int> order(vocab_size);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + kScoreBeamSize, order.end(),
        [&](int a, int b) { return logits_row[a] > logits_row[b]; });

    std::vector<std::pair<int, float>> candidates;
    for (int i = 0; i < kScoreBeamSize; ++i) {
      const int token = order[i];
      const float prob = static_cast<float>(
          std::exp(logits_row[token] - max_logit) / denom);
      if (prob > kMinTokenProb && searchable_token_ids_.count(token)) {
        candidates.emplace_back(token, prob);
      }
    }
    return candidates;
  }

  static void MaybeUpdateLastEmission(std::vector<EmittedToken>* emitted,
                                      int token, int frame, float prob) {
    if (!emitted->empty() && prob > emitted->back().best_prob) {
      emitted->back() = EmittedToken{token, frame, prob};
    }
  }

  // A: blank → 前缀不变
  static void ExpandBlank(const std::vector<int>& prefix, const PathState& src,
                          float prob, Beam* next) {
    auto& dst = (*next)[prefix];
    dst.blank_ending += src.Total() * prob;
    dst.emitted = src.emitted;
  }

  // B: 与末尾相同 → 叠在同一字，或经 blank 后再发一次
  static void ExpandRepeat(const std::vector<int>& prefix, const PathState& src,
                           int token, float prob, int frame, Beam* next) {
    if (std::abs(src.non_blank_ending) >= 1e-6) {
      auto& dst = (*next)[prefix];
      dst.non_blank_ending += src.non_blank_ending * prob;
      dst.emitted = src.emitted;
      MaybeUpdateLastEmission(&dst.emitted, token, frame, prob);
    }
    if (std::abs(src.blank_ending) >= 1e-6) {
      auto extended = prefix;
      extended.push_back(token);
      auto& dst = (*next)[extended];
      dst.non_blank_ending += src.blank_ending * prob;
      dst.emitted = src.emitted;
      dst.emitted.push_back(EmittedToken{token, frame, prob});
    }
  }

  // C: 新 token → 前缀加长
  static void ExpandNewToken(const std::vector<int>& prefix,
                             const PathState& src, int token, float prob,
                             int frame, Beam* next) {
    auto extended = prefix;
    extended.push_back(token);
    auto& dst = (*next)[extended];
    dst.non_blank_ending += src.Total() * prob;
    if (dst.emitted.empty()) {
      dst.emitted = src.emitted;
      dst.emitted.push_back(EmittedToken{token, frame, prob});
    } else {
      MaybeUpdateLastEmission(&dst.emitted, token, frame, prob);
    }
  }

  static void ExpandOneStep(const std::vector<int>& prefix,
                            const PathState& src, int token, float prob,
                            int frame, Beam* next) {
    const int last = prefix.empty() ? -1 : prefix.back();
    if (token == kBlankId) {
      ExpandBlank(prefix, src, prob, next);
    } else if (token == last) {
      ExpandRepeat(prefix, src, token, prob, frame, next);
    } else {
      ExpandNewToken(prefix, src, token, prob, frame, next);
    }
  }

  static Beam KeepTopPaths(Beam beam) {
    std::vector<std::pair<std::vector<int>, PathState>> ranked(beam.begin(),
                                                               beam.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.second.Total() > b.second.Total();
    });

    Beam kept;
    const int keep = std::min<int>(kPathBeamSize, ranked.size());
    for (int i = 0; i < keep; ++i) {
      kept.emplace(std::move(ranked[i].first), std::move(ranked[i].second));
    }
    return kept;
  }

  Beam RunCtcBeam(const float* logits, int num_frames, int vocab_size) const {
    Beam beam;
    beam[{}] = PathState{1.0, 0.0, {}};

    for (int frame = 0; frame < num_frames; ++frame) {
      const float* row = logits + static_cast<size_t>(frame) * vocab_size;
      const auto candidates = CollectFrameCandidates(row, vocab_size);
      if (candidates.empty()) continue;

      Beam next;
      for (const auto& [token, prob] : candidates) {
        for (const auto& [prefix, state] : beam) {
          ExpandOneStep(prefix, state, token, prob, frame, &next);
        }
      }
      beam = KeepTopPaths(std::move(next));
    }
    return beam;
  }

  Detection MatchKeywords(const Beam& beam) const {
    std::vector<std::pair<std::vector<int>, PathState>> ranked(beam.begin(),
                                                               beam.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.second.Total() > b.second.Total();
    });

    for (const auto& [prefix, state] : ranked) {
      if (prefix.size() != state.emitted.size()) continue;

      for (const auto& [name, target_ids] : keyword_token_ids_) {
        const auto match = std::search(prefix.begin(), prefix.end(),
                                       target_ids.begin(), target_ids.end());
        if (match == prefix.end()) continue;

        const size_t offset =
            static_cast<size_t>(std::distance(prefix.begin(), match));
        double score = 1.0;
        for (size_t i = 0; i < target_ids.size(); ++i) {
          score *= state.emitted[offset + i].best_prob;
        }
        return Detection{true, DisplayName(name),
                         static_cast<float>(std::sqrt(score))};
      }
    }
    return {};
  }

  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, std::vector<std::string>> lexicon_;
  std::map<std::string, std::vector<int>> keyword_token_ids_;
  std::set<int> searchable_token_ids_;
};

}  // namespace

// =============================================================================
// 3. KwsEngine：特征 → ONNX → 解码 → 阈值
// =============================================================================

struct KwsEngine::Impl {
  explicit Impl(const Options& options)
      : env(ORT_LOGGING_LEVEL_WARNING, "fsmn-kws"),
        session(nullptr),
        min_score(options.min_score),
        verbose(options.verbose) {
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::filesystem::path dir(options.model_dir);
    session = Ort::Session(env, (dir / "fsmn_kws.onnx").c_str(),
                           session_options);
    if (!features.LoadCmvn(dir / "am.mvn.dim80_l2r2") ||
        !decoder.Load(dir, options.keywords)) {
      throw std::runtime_error("加载 KWS 资源失败");
    }
  }

  Detection Infer(const std::vector<float>& waveform) {
    int num_frames = 0;
    std::vector<float> feats = features.Compute(waveform, &num_frames);
    if (num_frames <= 0) return {};

    const float* logits = nullptr;
    int logit_frames = 0;
    int vocab_size = 0;
    RunOnnx(feats, num_frames, &logits, &logit_frames, &vocab_size);

    return ApplyScoreThreshold(
        decoder.Decode(logits, logit_frames, vocab_size));
  }

  void RunOnnx(std::vector<float>& feats, int num_frames, const float** logits,
               int* logit_frames, int* vocab_size) {
    const std::array<int64_t, 3> shape{1, num_frames, kFeatureDim};
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(
        memory, feats.data(), feats.size(), shape.data(), shape.size());

    const char* input_names[] = {"speech"};
    const char* output_names[] = {"logits"};
    outs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                       output_names, 1);

    const auto shape_out = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape_out.size() != 3 || shape_out[2] != kVocabSize) {
      throw std::runtime_error("ONNX 输出维度错误");
    }
    *logits = outs[0].GetTensorData<float>();
    *logit_frames = static_cast<int>(shape_out[1]);
    *vocab_size = static_cast<int>(shape_out[2]);
  }

  Detection ApplyScoreThreshold(Detection result) const {
    if (result.detected && result.score < min_score) {
      if (verbose) {
        std::cout << "[debug] below threshold " << result.keyword << " "
                  << result.score << "\n";
      }
      return {};
    }
    if (verbose) {
      if (result.detected) {
        std::cout << "[debug] detected " << result.keyword << " "
                  << result.score << "\n";
      } else {
        std::cout << "[debug] rejected\n";
      }
    }
    return result;
  }

  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;
  std::vector<Ort::Value> outs;
  FeatureExtractor features;
  KeywordDecoder decoder;
  float min_score;
  bool verbose;
};

KwsEngine::KwsEngine(const Options& options)
    : impl_(std::make_unique<Impl>(options)) {}

KwsEngine::~KwsEngine() = default;

Detection KwsEngine::Infer(const std::vector<float>& waveform) {
  return impl_->Infer(waveform);
}
