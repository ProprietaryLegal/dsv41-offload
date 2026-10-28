#!/usr/bin/env bash
# FW3b (coordinator only, 2026-10-26): the sparkworker Embedded Controller (EC 0x02000005 -> 0x02000007, ec_3.5.11.4) and USB-PD
# (0x00000001 -> 0x00000516, usbpd_5.22) capsules. FW3 flashed the BSP (0x03000008, BIOS 0106) but the ESRT shows the EC
# unchanged (entry c3cccaf0… ver 0x02000005, last attempt 0x02000005): the BSP file name lists ec_3.5.11.4, the EC is a
# separate capsule. sparkhead received its EC and PD capsules in FW1. NDM 2026-10-26: "Ok update firmware".
# Everything below is FW3 with the target set changed; FW3's header follows for provenance.
# FW3 (coordinator only, 2026-10-26): sparkworker (rank 1, ASUS Ascent GX10) board firmware via fwupd/LVFS capsule-on-disk —
# the step FW2 skipped. FW2's "no capsules on the GX10" came from a fwupd probe that TIMED OUT ("Failed to connect to daemon",
# FW2/fwupd-updates-pre.txt), not from an empty offer; on 2026-10-26 ~21:30Z fwupd offers the ASUS BSP capsule OTA2607.1
# (UEFI Device Firmware 0x03000006 -> 0x03000008, bsp_ota2607.1_socfw308_bios0106_ec_3.5.11.4; OTA2604 notes list NCCL
# performance improvements), so sparkhead (NVIDIA OTA2607.1-era UEFI, FW1) and sparkworker run matching SoC firmware afterwards.
# Only the BSP capsule is flashed (it carries SoC fw 308, BIOS 0106 and EC 3.5.11.4); the separate EC and USB-PD offers are
# recorded, not flashed. No packages change: kernel 7.0.0-1019-nvidia / driver 580.178.04 / DGX OS 7.6.0 stay.
# Derived from fw2_sparkworker.sh (S43-checked pattern). Run on Controller, one step at a time, in order; each stops at its first failed gate:
#   fw3_sparkworker_firmware.sh pre      production serving: fwupd answers, the expected BSP release is offered, inventory
#   fw3_sparkworker_firmware.sh open     window.sh open (drain, HOLD, placeholders) then stop.sh (both ranks)
#   fw3_sparkworker_firmware.sh flash    stage the BSP capsule (detached, no reboot), confirm it is pending
#   fw3_sparkworker_firmware.sh reboot   reboot sparkworker, wait for a NEW boot (<= 45 min; capsules may reboot more than once), then stable
#   fw3_sparkworker_firmware.sh post     firmware now 0x03000008; kernel/driver/release unchanged; CX-7 links/RDMA; docker+GPU; logs
#   fw3_sparkworker_firmware.sh restore  W8/restore.sh start | check | close, six greedy IDENTICAL counted (S48)
# Rollback: none for firmware (downgrade is not offered as a routine path); a failed flash leaves the old firmware, and the
# post gates stop before production starts. Never power-cycle sparkworker during a flash.
set -euo pipefail
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
D=$C/FW3b; mkdir -p "$D"
S=sparkworker-wired                                 # ssh alias of sparkworker (the node being updated)
H=sparkhead                                   # production head: HOLD, window.sh, stop.sh
LOG=$D/fw3b.log
KERNEL=7.0.0-1019-nvidia DRIVER=580.178.04 RELEASE=7.6.0
BSP_FW=0x03000008
# name|old|new|cab — the exact set staged in one reboot
TARGETS="Embedded Controller|0x02000005|0x02000007|ec_3.5.11.4_signed.cab;UEFI Device Firmware|0x00000001|0x00000516|usbpd_5.22.cab"
LINKS="enp1s0f0np0=10.202.29.2/30 enp1s0f1np1=10.202.26.1/30 enP2p1s0f1np1=10.202.27.1/30"
PEERS="10.202.29.1 10.202.26.2 10.202.27.2"
die() { echo "FW3b STOP: $*" | tee -a "$LOG" >&2; exit 1; }
q() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
r() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$S" "$@"; }
h() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" "$@"; }
done_guard() { [ -e "$D/restore.done" ] && die "FW3b already restored ($D/restore.done)"; return 0; }
fwjson() { r 'sudo -n timeout 120 fwupdmgr get-devices --json --no-unreported-check 2>/dev/null'; }
# the BSP device = "UEFI Device Firmware" whose version is $1 (two devices share the name; the other is USB-PD 0x0000xxxx)
bsp_id() { python3 -c '
import json,sys
want=sys.argv[1]; d=json.load(sys.stdin)
ids=[x["DeviceId"] for x in d.get("Devices",[]) if x.get("Name")=="UEFI Device Firmware" and x.get("Version")==want]
print(ids[0] if len(ids)==1 else "")' "$1"; }
inventory() {
  r 'set +e; uname -r; grep -E "DGX_OTA_VERSION" /etc/dgx-release; nvidia-smi --query-gpu=driver_version --format=csv,noheader
     cat /sys/class/dmi/id/bios_version /sys/class/dmi/id/bios_date
     sudo -n timeout 120 fwupdmgr get-devices --no-unreported-check 2>/dev/null | grep -E "^[^ │├└]|Current version|Version:|Update State"
     grep -E "MemTotal|MemAvailable" /proc/meminfo; ip -br addr; ibv_devinfo 2>/dev/null | grep -E "hca_id|state"' > "$D/inventory-$1.txt" 2>&1
  q "inventory-$1 written ($(wc -l < "$D/inventory-$1.txt") lines)"
}
case "${1:-}" in
  pre)
    [ -e "$D/pre.done" ] && die "pre already done ($D/pre.done)"
    h 'test -f ~/dsv41-deploy/HOLD' && die "a HOLD exists on sparkhead; another window is open"
    [ -e "$C/FW2/restore.done" ] || die "FW2 has not been restored"
    r 'sudo -n timeout 180 fwupdmgr refresh --force >/dev/null 2>&1 || true'
    r 'sudo -n timeout 120 fwupdmgr get-updates --json --no-unreported-check 2>&1' > "$D/fwupd-updates-pre.json" \
      || die "fwupd get-updates failed or timed out (read $D/fwupd-updates-pre.json) — FW2's mistake: a failed probe is not an empty offer"
    [ -e "$C/FW3/restore.done" ] || die "FW3 (BSP) has not been restored and verified"
    python3 - "$D/fwupd-updates-pre.json" "$TARGETS" "$D/target-ids.txt" <<'PY2' | tee -a "$LOG" || die "the expected EC/PD releases are not offered exactly as planned"
import json, sys
d = json.load(open(sys.argv[1])); ids = []
for t in sys.argv[2].split(";"):
    name, old, new, cab = t.split("|")
    dev = [x for x in d.get("Devices", []) if x.get("Name") == name and x.get("Version") == old]
    assert len(dev) == 1, f"{len(dev)} devices {name} at {old}"
    rel = [r for r in dev[0].get("Releases", []) if r.get("Version") == new]
    assert rel and rel[0].get("Filename") == cab and "trusted-metadata" in rel[0].get("Flags", []), f"{name} {new} / {cab} not offered (trusted)"
    ids.append(dev[0]["DeviceId"]); print(f"OFFERED: {name} {old} -> {new} ({cab})")
others = [x["Name"] + " " + x.get("Version", "") for x in d.get("Devices", []) if x.get("Releases") and x["DeviceId"] not in ids]
assert not others, f"unplanned offers: {others}"
open(sys.argv[3], "w").write("\n".join(ids) + "\n")
PY2
    inventory pre
    q "PRE DONE: EC + USB-PD offered exactly as planned; no other offers"; touch "$D/pre.done" ;;
  open)
    [ -e "$D/pre.done" ] || die "run pre first"; done_guard
    bc=$(ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' 2>&1 | tail -2) || true; echo "$bc" >> "$LOG"
    grep -q "PRODUCTION HOLD OK" <<< "$bc" || die "x86host is not in PRODUCTION HOLD OK"
    h 'cd ~/dsv41-deploy && offload/launch/window.sh open "FW3b sparkworker ASUS EC 3.5.11.4 + USB-PD 5.22 firmware (coordinator 2026-10-26)"' | tee -a "$LOG" || die "window.sh open refused"
    h 'cd ~/dsv41-deploy && ./stop.sh </dev/null' >> "$LOG" 2>&1 || die "stop.sh failed"
    for n in spark sparkworker; do
      out=$(h "ssh -o BatchMode=yes $n 'docker ps --format {{.Names}} | grep -c dsv41 || true'"); [ "$out" = 0 ] || die "$n still runs a dsv41 container"
    done
    q "OPEN DONE: HOLD written, production stopped on both ranks"; touch "$D/open.done" ;;
  flash)
    [ -e "$D/open.done" ] || die "run open first"; done_guard
    : > "$D/fwupd-flash.log"
    for id in $(cat "$D/target-ids.txt"); do
      # detached from this ssh session: a dropped session must not interrupt fwupd while it writes a capsule to the ESP
      r "sudo -n rm -f /var/tmp/fw3b-fwupd.log; sudo -n systemd-run --unit=fw3b-fwupd --collect --wait -p StandardOutput=append:/var/tmp/fw3b-fwupd.log -p StandardError=append:/var/tmp/fw3b-fwupd.log timeout 600 fwupdmgr update $id -y --no-reboot-check --no-unreported-check" >> "$D/fwupd-flash.log" 2>&1 \
        || { r 'cat /var/tmp/fw3b-fwupd.log' >> "$D/fwupd-flash.log" 2>&1 || true; die "fwupd staging failed for $id (read $D/fwupd-flash.log); NOT rebooted; production may restart as is (restore)"; }
      r 'cat /var/tmp/fw3b-fwupd.log' >> "$D/fwupd-flash.log" 2>&1 || true
    done
    fwjson > "$D/devices-staged.json" || die "cannot read fwupd devices after staging"
    python3 - "$D/devices-staged.json" "$D/target-ids.txt" <<'PY2' | tee -a "$LOG" || die "an EC/PD update is not pending a reboot (read $D/fwupd-flash.log)"
import json, sys
d = json.load(open(sys.argv[1])); want = open(sys.argv[2]).read().split()
for i in want:
    x = [v for v in d["Devices"] if v["DeviceId"] == i][0]
    assert x.get("UpdateState") in (1, 4), f"{x['Name']}: update state {x.get('UpdateState')} (want 1 pending or 4 needs-reboot)"
    print(f"STAGED: {x['Name']} {x['Version']} update state {x.get('UpdateState')}")
PY2
    q "FLASH DONE: EC + USB-PD capsules staged on the ESP; reboot applies them"; touch "$D/flash.done" ;;
  reboot)
    [ -e "$D/flash.done" ] || die "run flash first"; done_guard
    old=$(r cat /proc/sys/kernel/random/boot_id) || die "cannot read boot_id"
    r 'sudo -n systemctl reboot' || q "reboot command returned non-zero (the session may have dropped); waiting for a NEW boot_id"
    q "REBOOT issued; waiting up to 45 min for a new boot (the capsule may reboot more than once; never power it off)"
    sleep 60; deadline=$((SECONDS + 2640))
    until b=$(r cat /proc/sys/kernel/random/boot_id 2>/dev/null) && [ -n "$b" ] && [ "$b" != "$old" ] || [ $SECONDS -ge $deadline ]; do sleep 30; done
    b=$(r cat /proc/sys/kernel/random/boot_id 2>/dev/null) && [ "$b" != "$old" ] || die "sparkworker not back on a NEW boot after 45 min: ask NDM to look at it (power LED); do NOT power-cycle during a firmware flash"
    # stable: the same boot for 3 minutes (a capsule that reboots again would change it)
    sleep 180; [ "$(r cat /proc/sys/kernel/random/boot_id 2>/dev/null)" = "$b" ] || die "sparkworker rebooted again within 3 min; rerun reboot's wait by hand before post"
    r 'tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold "exec -a \"vllm serve [DSV41-OFFLOAD hold]\" sleep 86400"'
    h 'test -f ~/dsv41-deploy/HOLD' || die "HOLD file missing on sparkhead after the sparkworker reboot"
    q "REBOOT DONE: sparkworker on a new, stable boot; hold placeholder re-created"; touch "$D/reboot.done" ;;
  post)
    [ -e "$D/reboot.done" ] || die "run reboot first"
    inventory post
    fwjson > "$D/devices-post.json" || die "cannot read fwupd devices after the reboot"
    python3 - "$D/devices-post.json" "$TARGETS" "$BSP_FW" <<'PY2' | tee -a "$LOG" || die "EC/PD/BSP versions are not as planned after the reboot (read $D/inventory-post.txt)"
import json, sys
d = json.load(open(sys.argv[1]))
have = {(x.get("Name"), x.get("Version")) for x in d["Devices"]}
for t in sys.argv[2].split(";"):
    name, old, new, cab = t.split("|")
    assert (name, new) in have, f"{name} is not {new}"
assert ("UEFI Device Firmware", sys.argv[3]) in have, "BSP no longer " + sys.argv[3]
print("VERSIONS: EC + USB-PD at target, BSP unchanged")
PY2
    [ "$(r uname -r)" = "$KERNEL" ] || die "kernel is $(r uname -r), not $KERNEL"
    r "grep -q 'DGX_OTA_VERSION=\"$RELEASE\"' /etc/dgx-release" || die "dgx-release is not $RELEASE"
    r 'grep -qw kho=off /proc/cmdline' || die "kho=off missing from the kernel command line"
    [ "$(r 'nvidia-smi --query-gpu=driver_version --format=csv,noheader')" = "$DRIVER" ] || die "nvidia-smi driver is not $DRIVER"
    for l in $LINKS; do r "ip -br addr show ${l%%=*} | grep -q 'UP .*${l#*=}'" || die "link ${l%%=*} is not UP with ${l#*=}"; done
    for p in $PEERS; do r "ping -c 3 -W 2 $p >/dev/null" || die "no ping to $p"; done
    [ "$(r 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')" -ge 4 ] || die "fewer than 4 RDMA ports PORT_ACTIVE"
    h 'ssh -o BatchMode=yes sparkworker true' || die "sparkhead cannot reach sparkworker (sparkworker) over the CX-7 link"
    IMAGE=$(h 'cd ~/dsv41-deploy && . ./dsv41.env >/dev/null 2>&1; echo $IMAGE')
    r "docker run --rm --pull never --gpus all --entrypoint nvidia-smi $IMAGE -L" > "$D/docker-gpu.txt" 2>&1 || die "docker GPU run failed (read $D/docker-gpu.txt)"
    r 'sudo -n journalctl -k -b --no-pager 2>/dev/null | grep -iE "NVRM: Xid|mlx5.*(error|fail)|r8127.*(error|fail)|BUG:|Oops" | head -20' > "$D/kernel-errors.txt" || true
    if [ -s "$D/kernel-errors.txt" ]; then
      [ "${FW3_ACCEPT_KLOG:-0}" = 1 ] || die "kernel log has GPU/NIC error lines (read $D/kernel-errors.txt; rerun post with FW3_ACCEPT_KLOG=1 if benign)"
      q "kernel log lines accepted after review: $(wc -l < "$D/kernel-errors.txt")"
    fi
    q "POST DONE: EC 0x02000007 + USB-PD 0x00000516, BSP $BSP_FW, BIOS $(r cat /sys/class/dmi/id/bios_version), kernel/driver/release unchanged, links + RDMA + docker GPU OK"
    touch "$D/post.done" ;;
  restore)
    [ -e "$D/open.done" ] || die "nothing to restore (open not run)"; done_guard
    bash $C/W8/restore.sh start | tee "$D/restore-start.txt" | tee -a "$LOG"
    grep -q "K1 START VERIFIED" "$D/restore-start.txt" || die "no K1 START VERIFIED (read $D/restore-start.txt)"
    bash $C/W8/restore.sh check | tee "$D/restore-check.txt" | tee -a "$LOG"
    [ "$(grep -cE 'greedy K1-post vs restored: [0-9]+ IDENTICAL [0-9]+$' "$D/restore-check.txt")" = 6 ] || die "not six greedy IDENTICAL lines (read $D/restore-check.txt); production stays in the window"
    grep -q "PRODUCTION HOLD OK" "$D/restore-check.txt" || die "no PRODUCTION HOLD OK"
    bash $C/W8/restore.sh close | tee "$D/restore-close.txt" | tee -a "$LOG"; grep -q "CLOSE: window closed" "$D/restore-close.txt" || die "window did not close"
    q "RESTORE DONE: K1 START VERIFIED, six greedy IDENTICAL, PRODUCTION HOLD OK, window closed"
    touch "$D/restore.done" ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
