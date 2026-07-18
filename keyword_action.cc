#include "keyword_action.h"

#include <chrono>
#include <iostream>
#include <utility>

#include "sound_tracking_action.h"

ConsoleKeywordAction::ConsoleKeywordAction(bool verbose) : verbose_(verbose) {}

void ConsoleKeywordAction::OnKeyword(const KeywordHit& hit) {
  std::cout << hit.keyword << "已识别\n";
  if (verbose_) {
    std::cout << "[debug] keyword=" << hit.keyword << " score=" << hit.score
              << "\n";
  }
}

void KeywordActionList::Add(std::unique_ptr<KeywordAction> action) {
  if (action) actions_.push_back(std::move(action));
}

void KeywordActionList::OnKeyword(const KeywordHit& hit) {
  for (auto& action : actions_) {
    action->OnKeyword(hit);
  }
}

void KeywordActionList::OnVoiceActivity(bool has_speech) {
  for (auto& action : actions_) {
    action->OnVoiceActivity(has_speech);
  }
}

std::unique_ptr<KeywordAction> CreateDefaultKeywordAction(
    const Options& options) {
  // 未来在这里追加其它动作即可，例如：
  //   list->Add(std::make_unique<GpioKeywordAction>(...));
  //   list->Add(std::make_unique<MqttKeywordAction>(...));
  auto list = std::make_unique<KeywordActionList>();
  list->Add(std::make_unique<ConsoleKeywordAction>(options.verbose));
  list->Add(std::make_unique<SoundTrackingKeywordAction>(
      std::chrono::seconds(options.session_timeout_seconds)));
  return list;
}
