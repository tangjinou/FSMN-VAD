#include "common.h"

#include <set>
#include <sstream>
#include <unordered_map>

std::vector<std::string> SplitCsv(const std::string& text) {
  std::stringstream ss(text);
  std::string item;
  std::vector<std::string> out;
  while (std::getline(ss, item, ',')) {
    const auto begin = item.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) continue;
    const auto end = item.find_last_not_of(" \t\r\n");
    out.push_back(item.substr(begin, end - begin + 1));
  }
  return out;
}

std::vector<std::string> Utf8Chars(const std::string& text) {
  std::vector<std::string> chars;
  for (size_t i = 0; i < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    size_t nbytes = 1;
    if ((c & 0xE0) == 0xC0) nbytes = 2;
    else if ((c & 0xF0) == 0xE0) nbytes = 3;
    else if ((c & 0xF8) == 0xF0) nbytes = 4;
    chars.push_back(text.substr(i, nbytes));
    i += nbytes;
  }
  return chars;
}

std::vector<std::string> ExpandKeywords(const std::string& csv) {
  // wǔ kōng 常落到「武/五/午/舞 + 空」
  static const std::unordered_map<std::string, std::vector<std::string>> kAlias{
      {"悟空", {"武空", "五空", "午空", "舞空"}},
  };

  std::vector<std::string> out;
  std::set<std::string> seen;
  auto add = [&](const std::string& w) {
    if (seen.insert(w).second) out.push_back(w);
  };

  for (const auto& word : SplitCsv(csv)) {
    add(word);
    if (const auto it = kAlias.find(word); it != kAlias.end()) {
      for (const auto& alias : it->second) add(alias);
    }
  }
  return out;
}

std::string DisplayName(const std::string& keyword) {
  static const std::set<std::string> kWukongAliases{"武空", "五空", "午空",
                                                    "舞空"};
  return kWukongAliases.count(keyword) ? "悟空" : keyword;
}
