// ofl_probe: read-only RDMA discovery. Prints every verbs device, its ports (state, link layer, active MTU, speed)
// and each port's GID table entries with type (IB / RoCE v1 / RoCE v2) and, for IPv4-mapped GIDs, the address.
// Opens devices and queries them; creates nothing. Used to pick (device, port, GID index) for ofl configs.
// Build: gcc -O2 ofl_probe.c -o ofl_probe -libverbs   (or the full path of libibverbs.so.1)
#include <infiniband/verbs.h>
#include <stdio.h>
#include <string.h>

static const char* gid_type(uint32_t t) {
  switch (t) {
    case IBV_GID_TYPE_IB: return "IB";
    case IBV_GID_TYPE_ROCE_V1: return "RoCEv1";
    case IBV_GID_TYPE_ROCE_V2: return "RoCEv2";
    default: return "?";
  }
}

int main(void) {
  int n = 0;
  struct ibv_device** list = ibv_get_device_list(&n);
  if (!list) { perror("ibv_get_device_list"); return 1; }
  for (int i = 0; i < n; i++) {
    struct ibv_context* c = ibv_open_device(list[i]);
    if (!c) { printf("%s: open failed\n", ibv_get_device_name(list[i])); continue; }
    struct ibv_device_attr da;
    memset(&da, 0, sizeof(da));
    ibv_query_device(c, &da);
    printf("%s fw=%s ports=%d max_qp_wr=%d max_sge=%d max_mr_size=%llu\n", ibv_get_device_name(list[i]), da.fw_ver,
           da.phys_port_cnt, da.max_qp_wr, da.max_sge, (unsigned long long)da.max_mr_size);
    for (int p = 1; p <= da.phys_port_cnt; p++) {
      struct ibv_port_attr pa;
      if (ibv_query_port(c, p, &pa)) { printf("  port %d: query failed\n", p); continue; }
      printf("  port %d state=%s link=%s active_mtu=%d(%d B) speed=%u width=%u gid_tbl_len=%d\n", p,
             ibv_port_state_str(pa.state), pa.link_layer == IBV_LINK_LAYER_ETHERNET ? "Ethernet" : "IB",
             pa.active_mtu, 128 << pa.active_mtu, pa.active_speed, pa.active_width, pa.gid_tbl_len);
      for (int g = 0; g < pa.gid_tbl_len; g++) {
        struct ibv_gid_entry e;
        if (ibv_query_gid_ex(c, p, g, &e, 0)) continue;  // empty slot
        int zero = 1;
        for (int k = 0; k < 16; k++) zero &= e.gid.raw[k] == 0;
        if (zero) continue;
        printf("    gid %d %-6s ndev_ifindex=%u ", g, gid_type(e.gid_type), e.ndev_ifindex);
        for (int k = 0; k < 16; k += 2) printf("%02x%02x%s", e.gid.raw[k], e.gid.raw[k + 1], k < 14 ? ":" : "");
        int v4 = 1;
        for (int k = 0; k < 10; k++) v4 &= e.gid.raw[k] == 0;
        v4 &= e.gid.raw[10] == 0xff && e.gid.raw[11] == 0xff;
        if (v4) printf("  ipv4=%u.%u.%u.%u", e.gid.raw[12], e.gid.raw[13], e.gid.raw[14], e.gid.raw[15]);
        printf("\n");
      }
    }
    ibv_close_device(c);
  }
  ibv_free_device_list(list);
  return 0;
}
