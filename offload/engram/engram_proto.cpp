// engram_proto implementation: identity, safetensors header + manifest parsing (a small JSON reader), SHA-256.
#include "engram_proto.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>

namespace engram {

namespace {

// ---- a small JSON reader (objects, arrays, strings, integers / numbers, true/false/null) -------------------------
struct J {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  bool b = false;
  bool is_int = false;
  long long i = 0;
  double d = 0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;
  const J* get(const std::string& k) const {
    for (const auto& kv : o)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

struct Parser {
  const char* p;
  const char* e;
  std::string err;
  void ws() {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
  }
  bool fail(const char* m) {
    if (err.empty()) err = m;
    return false;
  }
  bool str(std::string& out) {
    if (p >= e || *p != '"') return fail("expected string");
    p++;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        if (++p >= e) return fail("bad escape");
        switch (*p) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            if (e - p < 5) return fail("bad \\u escape");
            unsigned v = (unsigned)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16);
            if (v < 0x80) out += (char)v;
            else if (v < 0x800) { out += (char)(0xC0 | (v >> 6)); out += (char)(0x80 | (v & 0x3F)); }
            else { out += (char)(0xE0 | (v >> 12)); out += (char)(0x80 | ((v >> 6) & 0x3F)); out += (char)(0x80 | (v & 0x3F)); }
            p += 4;
            break;
          }
          default: return fail("bad escape");
        }
        p++;
      } else {
        out += *p++;
      }
    }
    if (p >= e) return fail("unterminated string");
    p++;
    return true;
  }
  bool val(J& v, int depth) {
    if (depth > 64) return fail("nesting too deep");
    ws();
    if (p >= e) return fail("unexpected end");
    if (*p == '{') {
      v.t = J::Obj;
      p++;
      ws();
      if (p < e && *p == '}') { p++; return true; }
      for (;;) {
        ws();
        std::string k;
        if (!str(k)) return false;
        ws();
        if (p >= e || *p != ':') return fail("expected ':'");
        p++;
        J c;
        if (!val(c, depth + 1)) return false;
        v.o.emplace_back(std::move(k), std::move(c));
        ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; return true; }
        return fail("expected ',' or '}'");
      }
    }
    if (*p == '[') {
      v.t = J::Arr;
      p++;
      ws();
      if (p < e && *p == ']') { p++; return true; }
      for (;;) {
        J c;
        if (!val(c, depth + 1)) return false;
        v.a.push_back(std::move(c));
        ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; return true; }
        return fail("expected ',' or ']'");
      }
    }
    if (*p == '"') { v.t = J::Str; return str(v.s); }
    if (e - p >= 4 && !strncmp(p, "true", 4)) { v.t = J::Bool; v.b = true; p += 4; return true; }
    if (e - p >= 5 && !strncmp(p, "false", 5)) { v.t = J::Bool; p += 5; return true; }
    if (e - p >= 4 && !strncmp(p, "null", 4)) { p += 4; return true; }
    const char* s = p;
    if (p < e && (*p == '-' || *p == '+')) p++;
    bool frac = false;
    while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
      if (*p == '.' || *p == 'e' || *p == 'E') frac = true;
      p++;
    }
    if (p == s) return fail("unexpected character");
    std::string num(s, p);
    v.t = J::Num;
    v.d = strtod(num.c_str(), nullptr);
    if (!frac) { v.is_int = true; v.i = strtoll(num.c_str(), nullptr, 10); }
    return true;
  }
};

bool parse_json(const char* p, size_t n, J& out, std::string& msg) {
  Parser ps{p, p + n, {}};
  if (!ps.val(out, 0)) { msg = "JSON: " + ps.err; return false; }
  return true;
}

bool read_file(const std::string& path, std::string& out, std::string& msg) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { msg = "cannot open " + path + ": " + strerror(errno); return false; }
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, k);
  fclose(f);
  return true;
}

std::string basename_of(const std::string& p) {
  const size_t k = p.rfind('/');
  return k == std::string::npos ? p : p.substr(k + 1);
}

// "layers.<N>.engram.embed.<suffix>" -> N, else -1
int engram_layer(const std::string& name, const char* suffix) {
  int n = -1, used = 0;
  char tail[64] = {0};
  if (sscanf(name.c_str(), "layers.%d.engram.embed.%63s%n", &n, tail, &used) != 2) return -1;
  if (strcmp(tail, suffix) != 0 || (size_t)used != name.size()) return -1;
  return n;
}

const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void block(uint32_t h[8], const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; i++) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; i++) {
    const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
    const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = S0 + mj;
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

}  // namespace

Sha256::Sha256() {
  const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  memcpy(h, init, sizeof(h));
}

void Sha256::update(const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  len += n;
  if (fill) {
    const size_t k = std::min(n, 64 - fill);
    memcpy(buf + fill, p, k);
    fill += k; p += k; n -= k;
    if (fill == 64) { block(h, buf); fill = 0; }
  }
  for (; n >= 64; p += 64, n -= 64) block(h, p);
  if (n) { memcpy(buf, p, n); fill = n; }
}

void Sha256::final(uint8_t out[32]) {
  const uint64_t bits = len * 8;
  const uint8_t pad = 0x80;
  update(&pad, 1);
  const uint8_t zero = 0;
  while (fill != 56) update(&zero, 1);
  uint8_t lb[8];
  for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
  update(lb, 8);
  for (int i = 0; i < 8; i++) {
    out[4 * i] = (uint8_t)(h[i] >> 24); out[4 * i + 1] = (uint8_t)(h[i] >> 16);
    out[4 * i + 2] = (uint8_t)(h[i] >> 8); out[4 * i + 3] = (uint8_t)h[i];
  }
}

std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; i++) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
  return s;
}

std::string identity_string(const TableDesc& d) {
  auto shape = [](const std::vector<uint64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); i++) o += (i ? "," : "") + std::to_string(s[i]);
    return o + "]";
  };
  auto tensor = [&](const TensorInfo& t) {
    return t.name + " " + t.dtype + " " + shape(t.shape) + " offset=" + std::to_string(t.abs_offset) +
           " bytes=" + std::to_string(t.nbytes);
  };
  return "engram-table-v1\nshard=" + d.shard + "\nshard_size=" + std::to_string(d.shard_size) +
         "\nshard_sha256=" + d.shard_sha256 + "\nlayer=" + std::to_string(d.layer_id) + "\nweight=" + tensor(d.weight) +
         "\nscale=" + tensor(d.scale) +
         "\nscale_layout=ue8m0 exponent byte per 32 weights, row-major [rows, 8]"
         "\nrow_layout=n*256 weight bytes then n*8 scale bytes\n";
}

void identity_digest(const TableDesc& d, uint8_t out[32]) {
  const std::string s = identity_string(d);
  Sha256 h;
  h.update(s.data(), s.size());
  h.final(out);
}

bool read_safetensors_header(const std::string& path, std::vector<TensorInfo>& out, std::string& msg) {
  // O_DIRECT (no page cache on the Sparks); a file system that refuses it gets a plain read of the header bytes
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  const bool direct = fd >= 0;
  if (!direct) fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) { msg = "cannot open " + path + ": " + strerror(errno); return false; }
  size_t cap = 1 << 16;
  void* raw = nullptr;
  if (posix_memalign(&raw, 4096, cap) != 0) { ::close(fd); msg = "out of memory"; return false; }
  std::unique_ptr<uint8_t, decltype(&free)> buf(static_cast<uint8_t*>(raw), &free);
  ssize_t got = pread(fd, buf.get(), cap, 0);
  if (got < 0 && direct && errno == EINVAL) {  // O_DIRECT open accepted but reads refused
    ::close(fd);
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    got = fd >= 0 ? pread(fd, buf.get(), cap, 0) : -1;
  }
  if (got < 8) { msg = "cannot read the header of " + path + ": " + strerror(errno); if (fd >= 0) ::close(fd); return false; }
  uint64_t hl = 0;
  memcpy(&hl, buf.get(), 8);
  if (hl == 0 || hl > (100u << 20)) { ::close(fd); msg = "implausible safetensors header length in " + path; return false; }
  if (8 + hl > (uint64_t)got) {
    const size_t need = ((8 + hl + 4095) / 4096) * 4096;
    void* r2 = nullptr;
    if (posix_memalign(&r2, 4096, need) != 0) { ::close(fd); msg = "out of memory"; return false; }
    buf.reset(static_cast<uint8_t*>(r2));
    got = pread(fd, buf.get(), need, 0);
    if (got < (ssize_t)(8 + hl)) { ::close(fd); msg = "short header read of " + path; return false; }
  }
  ::close(fd);
  J root;
  if (!parse_json(reinterpret_cast<const char*>(buf.get()) + 8, hl, root, msg)) { msg = path + ": " + msg; return false; }
  if (root.t != J::Obj) { msg = path + ": header is not an object"; return false; }
  for (const auto& kv : root.o) {
    if (kv.first == "__metadata__") continue;
    const J& t = kv.second;
    const J* dt = t.get("dtype");
    const J* sh = t.get("shape");
    const J* of = t.get("data_offsets");
    if (!dt || dt->t != J::Str || !sh || sh->t != J::Arr || !of || of->t != J::Arr || of->a.size() != 2) {
      msg = path + ": malformed tensor entry " + kv.first;
      return false;
    }
    TensorInfo ti;
    ti.name = kv.first;
    ti.dtype = dt->s;
    for (const auto& x : sh->a) {
      if (!x.is_int || x.i < 0) { msg = path + ": bad shape of " + kv.first; return false; }
      ti.shape.push_back((uint64_t)x.i);
    }
    if (!of->a[0].is_int || !of->a[1].is_int || of->a[1].i < of->a[0].i) { msg = path + ": bad offsets of " + kv.first; return false; }
    ti.abs_offset = 8 + hl + (uint64_t)of->a[0].i;
    ti.nbytes = (uint64_t)(of->a[1].i - of->a[0].i);
    out.push_back(std::move(ti));
  }
  return true;
}

bool manifest_entry(const std::string& manifest_path, const std::string& name, uint64_t& size, std::string& sha256,
                    std::string& msg) {
  std::string text;
  if (!read_file(manifest_path, text, msg)) return false;
  J root;
  if (!parse_json(text.data(), text.size(), root, msg)) { msg = manifest_path + ": " + msg; return false; }
  const J* e = root.get(name);
  if (!e || e->t != J::Obj) { msg = manifest_path + " has no entry for " + name; return false; }
  const J* sz = e->get("size");
  const J* h = e->get("sha256");
  if (!sz || !sz->is_int || !h || h->t != J::Str || h->s.size() != 64) {
    msg = manifest_path + ": entry " + name + " lacks size/sha256";
    return false;
  }
  size = (uint64_t)sz->i;
  sha256 = h->s;
  return true;
}

bool describe_table(const std::string& path, int64_t weight_offset, int64_t scale_offset,
                    const std::string& manifest_path, TableDesc& d, std::string& msg) {
  std::vector<TensorInfo> ts;
  if (!read_safetensors_header(path, ts, msg)) return false;
  const TensorInfo* w = nullptr;
  for (const auto& t : ts) {
    const int l = engram_layer(t.name, "weight");
    if (l < 0) continue;
    if (weight_offset >= 0 && t.abs_offset != (uint64_t)weight_offset) continue;
    if (w) { msg = path + ": more than one Engram weight tensor; name the offset"; return false; }
    w = &t;
    d.layer_id = l;
  }
  if (!w) {
    msg = path + (weight_offset >= 0 ? ": no Engram weight tensor at offset " + std::to_string(weight_offset)
                                     : std::string(": no Engram weight tensor"));
    return false;
  }
  const std::string sname = "layers." + std::to_string(d.layer_id) + ".engram.embed.scale";
  const TensorInfo* s = nullptr;
  for (const auto& t : ts)
    if (t.name == sname) s = &t;
  if (!s) { msg = path + ": no tensor " + sname; return false; }
  if (scale_offset >= 0 && s->abs_offset != (uint64_t)scale_offset) {
    msg = path + ": " + sname + " is at " + std::to_string(s->abs_offset) + ", not " + std::to_string(scale_offset);
    return false;
  }
  if (w->dtype != "F8_E4M3" || w->shape.size() != 2 || w->shape[1] != kRowBytes || w->nbytes != w->shape[0] * kRowBytes) {
    msg = path + ": " + w->name + " is not F8_E4M3 [rows, 256]";
    return false;
  }
  if (s->dtype != "F8_E8M0" || s->shape.size() != 2 || s->shape[0] != w->shape[0] || s->shape[1] != kScaleBytes ||
      s->nbytes != s->shape[0] * kScaleBytes) {
    msg = path + ": " + s->name + " is not F8_E8M0 [rows, 8] matching the weights";
    return false;
  }
  d.weight = *w;
  d.scale = *s;
  d.shard = basename_of(path);
  if (!manifest_entry(manifest_path, d.shard, d.shard_size, d.shard_sha256, msg)) return false;
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) { msg = "stat " + path + ": " + strerror(errno); return false; }
  if ((uint64_t)st.st_size != d.shard_size) {
    msg = path + " is " + std::to_string(st.st_size) + " bytes, the manifest says " + std::to_string(d.shard_size);
    return false;
  }
  return true;
}

}  // namespace engram
