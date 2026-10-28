// SPDX-License-Identifier: Apache-2.0
// X1 remote adaptation of HeNryous 0078 (afe4d5e). No transport or threading here.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace x1 {
class RowLookahead {
 public:
  static constexpr size_t kBytes = 264;
  static constexpr size_t kMaxRows = 8192;
  static constexpr size_t kBurst = 512;
  using Row = std::array<uint8_t, kBytes>;
  // All methods called with the owning table's la_mu_; one pending, replaceable snapshot.
  void replace(const std::vector<int64_t>& ids, int64_t bound) {
    if (ids.size() > kMaxRows) throw std::invalid_argument("lookahead rows exceed 8192");
    for (auto id : ids) if (id < 0 || id >= bound) throw std::invalid_argument("lookahead row out of range");
    ++version_;
    pending_.clear();
    std::unordered_set<int64_t> seen;
    for (auto id : ids) if (!rows_.count(id) && seen.insert(id).second) pending_.push_back(id);
  }
  uint64_t version() const { return version_; }
  bool pending() const { return !pending_.empty(); }
  std::vector<int64_t> take() {
    std::vector<int64_t> out;
    while (!pending_.empty() && out.size() < kBurst) {
      out.push_back(pending_.front()); pending_.pop_front();
    }
    return out;
  }
  void clear() { ++version_; pending_.clear(); rows_.clear(); order_.clear(); }
  bool copy(const int64_t* ids, uint8_t* w, uint8_t* s, size_t n) {
    for (size_t i=0; i<n; ++i) if (!rows_.count(ids[i])) return false; // no partial publication
    for (size_t i=0; i<n; ++i) {
      auto& item = rows_.at(ids[i]);
      std::memcpy(w+i*256, item.row.data(), 256);
      std::memcpy(s+i*8, item.row.data()+256, 8);
      order_.splice(order_.end(), order_, item.order);
    }
    return true;
  }
  // Foreground partial hits are useful even when a 4096-token chunk exceeds the cache.
  // Missing rows are fetched compactly into the PRIVATE bounce destination, then scattered by index.
  void split(const int64_t* ids, uint8_t* w, uint8_t* s, size_t n,
             std::vector<int64_t>& misses, std::vector<size_t>& indices) {
    misses.clear(); indices.clear();
    for (size_t i=0; i<n; ++i) {
      if (!copy(ids+i,w+i*256,s+i*8,1)) { misses.push_back(ids[i]); indices.push_back(i); }
    }
  }
  bool publish(uint64_t version, const std::vector<int64_t>& ids, const uint8_t* w, const uint8_t* s) {
    if (version != version_) return false; // cancel/restart/reconnect while RPC was in flight
    for (size_t i=0; i<ids.size(); ++i) {
      auto hit = rows_.find(ids[i]);
      if (hit == rows_.end()) {
        if (rows_.size() == kMaxRows) { rows_.erase(order_.front()); order_.pop_front(); }
        order_.push_back(ids[i]);
        Item item{}; item.order=std::prev(order_.end());
        hit=rows_.emplace(ids[i], item).first;
      }
      std::memcpy(hit->second.row.data(), w+i*256, 256);
      std::memcpy(hit->second.row.data()+256, s+i*8, 8);
    }
    return true;
  }
  size_t size() const { return rows_.size(); }
 private:
  struct Item { Row row; std::list<int64_t>::iterator order; };
  uint64_t version_=0;
  std::deque<int64_t> pending_;
  std::list<int64_t> order_;
  std::unordered_map<int64_t, Item> rows_;
};
} // namespace x1
