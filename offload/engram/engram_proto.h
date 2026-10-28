// engram_proto: the Engram row-lookup service contract ("engram-v1" over the ofl transport), shared by the x86host
// table server (offload/engram/server) and the Spark-side EngramRemoteTable (offload/engram/spark).
//
// A table = one layer's Engram embedding in one checkpoint shard: an FP8 E4M3 weight tensor [rows, 256] and its
// ue8m0 (F8_E8M0) scale tensor [rows, 8] (one scale per 32 weights). A row is 256 + 8 bytes; the engine's staging
// layout for n rows is the n weight rows (n * 256 B) followed by the n scale rows (n * 8 B).
//
// Identity: sha256 of a canonical text naming the shard (file name, size, sha256 from the repo's manifest.json),
// both tensors (name, dtype, shape, absolute byte offset, byte size) and the scale layout. The server computes it from
// the shard it loaded (and verifies the shard's sha256 while loading); the client computes it from its local shard's
// header + the manifest. OPEN and every LOOKUP carry it; a mismatch is refused, so a client can never read rows of
// another checkpoint or layer. Ops:
//   OPEN   (1): OpenReq -> OpenResp (a handle for (table, row_start, num_rows))
//   LOOKUP (2): LookupHdr + n uint32 row ids relative to row_start (kZeroRow = a row that reads as zeros, which is
//               what the disk reader returns for a negative / out-of-range id) -> n*256 weight bytes, n*8 scale bytes
//   STATS  (3): -> JSON text
#ifndef ENGRAM_PROTO_H
#define ENGRAM_PROTO_H

#include <stdint.h>

#include <string>
#include <vector>

namespace engram {

constexpr const char* kService = "engram-v1";
constexpr uint32_t kProtoVersion = 1;
constexpr uint16_t kOpOpen = 1, kOpLookup = 2, kOpStats = 3;
constexpr uint32_t kZeroRow = 0xFFFFFFFFu;
constexpr uint32_t kRowBytes = 256, kScaleBytes = 8;
constexpr uint16_t kDefaultPort = 18610;

// status codes a handler returns (negative = refused)
enum {
  kErrBadRequest = -100,   // malformed payload
  kErrIdentity = -101,     // identity does not match any loaded table / the handle's table
  kErrRange = -102,        // row range outside the table, or row/scale sizes differ
  kErrHandle = -103,       // unknown handle
  kErrRowId = -104,        // a row id >= num_rows that is not kZeroRow
  kErrTooMany = -105,      // more rows than the response capacity
  kErrNotReady = -106,     // tables still loading
};

#pragma pack(push, 1)
struct OpenReq {
  uint32_t version;
  uint32_t layer_id;
  uint64_t row_start;
  uint64_t num_rows;
  uint32_t row_bytes;
  uint32_t scale_bytes;
  uint8_t identity[32];
  char client[64];
};
struct OpenResp {
  uint32_t version;
  uint32_t handle;
  uint64_t table_rows;
  uint8_t identity[32];
  char desc[192];
};
struct LookupHdr {
  uint32_t handle;
  uint32_t n;
  uint8_t identity[32];
};
#pragma pack(pop)
static_assert(sizeof(OpenReq) == 128, "OpenReq layout");
static_assert(sizeof(OpenResp) == 240, "OpenResp layout");
static_assert(sizeof(LookupHdr) == 40, "LookupHdr layout");

struct TensorInfo {
  std::string name, dtype;
  std::vector<uint64_t> shape;
  uint64_t abs_offset = 0;  // from the start of the file
  uint64_t nbytes = 0;
};

struct TableDesc {
  std::string shard;         // file name (basename)
  std::string shard_sha256;  // hex, from the manifest
  uint64_t shard_size = 0;   // bytes, from the manifest (checked against the file)
  int layer_id = -1;
  TensorInfo weight, scale;
  uint64_t rows() const { return weight.shape.empty() ? 0 : weight.shape[0]; }
};

// The canonical identity text and its sha256.
std::string identity_string(const TableDesc& d);
void identity_digest(const TableDesc& d, uint8_t out[32]);
std::string hex(const uint8_t* p, size_t n);

// Tensors of a safetensors file (header read with O_DIRECT when the file system allows it, else a plain read of
// the first bytes). false + msg on error.
bool read_safetensors_header(const std::string& path, std::vector<TensorInfo>& out, std::string& msg);
// manifest.json entry {"size": N, "sha256": "..."} of a file name. false + msg if missing.
bool manifest_entry(const std::string& manifest_path, const std::string& name, uint64_t& size, std::string& sha256,
                    std::string& msg);
// Describe the Engram table of `path`: the layer whose weight tensor starts at `weight_offset` (or, with
// weight_offset < 0, the only layers.N.engram.embed.weight in the file), its scale tensor, dtype/shape checks,
// the manifest sha256/size, and the file size check. false + msg on any mismatch.
bool describe_table(const std::string& path, int64_t weight_offset, int64_t scale_offset,
                    const std::string& manifest_path, TableDesc& d, std::string& msg);

// SHA-256 (portable implementation; the server hashes whole shards with OpenSSL when it is there).
struct Sha256 {
  uint32_t h[8];
  uint64_t len = 0;
  uint8_t buf[64];
  size_t fill = 0;
  Sha256();
  void update(const void* p, size_t n);
  void final(uint8_t out[32]);
};

}  // namespace engram
#endif
