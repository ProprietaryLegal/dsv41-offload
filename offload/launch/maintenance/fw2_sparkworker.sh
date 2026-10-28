#!/usr/bin/env bash
# FW2 (coordinator only, 2026-10-26): sparkworker (rank 1, ASUS Ascent GX10) to the same DGX OS 7.6.0 / kernel
# 7.0.0-1019-nvidia / driver 580.178.04 package set sparkhead received in FW1 (plus the CX-7 firmware burn by
# nvidia-spark-mlnx-firmware-manager 6.0.1), so both ends of the Spark links match. No board firmware: fwupd offers none to
# the GX10 and NVIDIA's October firmware is Founders Edition only. Derived from fw1_sparkhead.sh (S43-checked); production
# control (HOLD, stop/start) stays on sparkhead. Runbook: documents/plans/2026-10-26-fw1-sparkhead-update.md §FW2.
# Run on Controller, one step at a time, in order; every step stops at its first failed gate:
#   fw2_sparkworker.sh pre            production serving: inventory, /etc backup, repacked rollback debs, upgrade simulation
#   fw2_sparkworker.sh open           window.sh open (drain, HOLD, placeholders) then stop.sh (both ranks)
#   fw2_sparkworker.sh upgrade        apt dist-upgrade (keep local config files), pre-reboot gates (no firmware capsules)
#   fw2_sparkworker.sh reboot         reboot sparkworker, wait for ssh (<= 45 min), re-create the hold placeholder
#   fw2_sparkworker.sh post           platform gates: kernel, release, driver, CX-7 links/RDMA, docker+GPU, memory, logs
#   fw2_sparkworker.sh restore        W8/restore.sh start | check | close with all four strings required
set -euo pipefail
C=/opt/dsv41/campaigns/DSV41-OFFLOAD
D=$C/FW2; mkdir -p "$D"
S=sparkworker-wired                                 # ssh alias of sparkworker (the node being updated)
H=sparkhead                                   # production head: HOLD, window.sh, stop.sh, the CX-7 view of sparkworker (sparkworker)
LOG=$D/fw2.log
KERNEL=7.0.0-1019-nvidia DRIVER=580.178.04 RELEASE=7.6.0
OLD_KERNEL=6.17.0-1026-nvidia
# CX-7 addresses that must come back unchanged (sparkworker side) and their peers
LINKS="enp1s0f0np0=10.202.29.2/30 enp1s0f1np1=10.202.26.1/30 enP2p1s0f1np1=10.202.27.1/30"
PEERS="10.202.29.1 10.202.26.2 10.202.27.2"  # x86host Engram/drafter link; sparkhead x2
# The packages whose installed versions are repacked before the upgrade so a 6.17/580.159 rollback never depends on the archive
ROLLBACK_PKGS="nvidia-driver-580-open nvidia-kernel-common-580 nvidia-kernel-source-580-open libnvidia-compute-580
 libnvidia-common-580 libnvidia-extra-580 nvidia-utils-580 nvidia-compute-utils-580 linux-modules-nvidia-580-open-$OLD_KERNEL
 linux-modules-nvidia-580-open-nvidia-hwe-24.04 nvidia-container-toolkit nvidia-container-toolkit-base docker-ce docker-ce-cli
 containerd.io dgx-release libnvidia-gl-580 libnvidia-decode-580 libnvidia-encode-580 libnvidia-cfg1-580 libnvidia-fbc1-580
 xserver-xorg-video-nvidia-580 nvidia-firmware-580-580.159.03 libnvidia-container1 libnvidia-container-tools"
die() { echo "FW2 STOP: $*" | tee -a "$LOG" >&2; exit 1; }
q() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
r() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$S" "$@"; }
h() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" "$@"; }
done_guard() { [ -e "$D/restore.done" ] && die "FW2 already restored ($D/restore.done); a new run needs a new FW1 directory"; return 0; }
meminfo() { r 'grep -E "MemTotal|MemAvailable|SwapFree" /proc/meminfo' > "$D/meminfo-$1.txt"; q "meminfo-$1: $(tr -s ' ' < "$D/meminfo-$1.txt" | tr '\n' ' ')"; }
inventory() {  # $1 = pre|post
  r 'set +e; uname -r; grep -E "DGX_OTA_VERSION|DGX_SWBUILD_VERSION" /etc/dgx-release; nvidia-smi --query-gpu=driver_version --format=csv,noheader
     sudo -n fwupdmgr get-devices --no-unreported-check 2>/dev/null | grep -E "^[^ │├└]|Current version|Version:" ; grep -E "MemTotal|MemAvailable|SwapTotal" /proc/meminfo
     for i in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1 enP7s7; do printf "%s %s %s\n" $i "$(ethtool -i $i 2>/dev/null | grep firmware-version)" "$(cat /sys/class/net/$i/operstate)"; done
     ip -br addr; ibv_devinfo 2>/dev/null | grep -E "hca_id|state"; lsmod | grep -E "^(nvidia|mlx5|r8127)"
     dpkg -l | awk "/^ii/{print \$2, \$3}" | grep -E "^(linux-image|linux-modules-nvidia|nvidia-driver|docker-ce |nvidia-container-toolkit |dgx-release|nvidia-spark-mlnx)"
     sudo -n grep -E "menuentry .DGX" /boot/grub/grub.cfg | sed "s/--class.*//"; systemctl --failed --no-legend --no-pager' > "$D/inventory-$1.txt" 2>&1
  q "inventory-$1 written ($(wc -l < "$D/inventory-$1.txt") lines)"
}
case "${1:-}" in
  pre)
    [ -e "$D/pre.done" ] && die "pre already done ($D/pre.done)"
    h 'test -f ~/dsv41-deploy/HOLD' && die "a HOLD exists on sparkhead; another window is open"
    [ -e "$C/FW1/restore.done" ] || die "FW1 (sparkhead) has not been restored and verified; FW2 comes after it"
    inventory pre; meminfo pre-serving
    r 'set -e; mkdir -p ~/fw1/rollback-debs; { sudo -n tar czf ~/fw1/etc-pre.tgz /etc 2>/dev/null || [ $? -eq 1 ]; }; sudo -n chmod 600 ~/fw1/etc-pre.tgz; dpkg --get-selections > ~/fw1/selections-pre.txt
       apt-mark showmanual > ~/fw1/manual-pre.txt; sudo -n cp /boot/grub/grub.cfg ~/fw1/grub-pre.cfg; sudo -n chown $USER ~/fw1/grub-pre.cfg' \
      || die "backup on sparkworker failed"
    r 'command -v dpkg-repack >/dev/null || sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -q dpkg-repack >/dev/null' || die "dpkg-repack unavailable"
    r "set -e; cd ~/fw1/rollback-debs; for p in $(echo $ROLLBACK_PKGS); do dpkg -s \$p >/dev/null 2>&1 || { echo \"not installed: \$p\"; continue; }; sudo -n dpkg-repack \$p >/dev/null 2>&1 || { echo \"REPACK FAILED: \$p\"; exit 1; }; done; sudo -n chown -R \$USER .; ls -1 | wc -l" \
      | tee -a "$LOG" || die "rollback repack failed"
    # the /etc tarball holds /etc/shadow: it stays on sparkworker (~/fw1, mode 600); only package lists and grub.cfg come here
    scp -q -o BatchMode=yes "$S:fw1/selections-pre.txt" "$S:fw1/manual-pre.txt" "$S:fw1/grub-pre.cfg" "$D/" || die "copy of backups failed"
    mkdir -p "$D/rollback-debs"; scp -q -o BatchMode=yes "$S:fw1/rollback-debs/*.deb" "$D/rollback-debs/" || die "copy of rollback debs failed"
    r 'sudo -n apt-get update -qq && apt-get -s -o Dpkg::Options::=--force-confold dist-upgrade' > "$D/simulate-pre.txt" 2>&1 || die "simulation failed"
    grep -q "^Inst linux-image-$KERNEL" "$D/simulate-pre.txt" || die "simulation does not install linux-image-$KERNEL"
    grep -q "^Inst nvidia-driver-580-open .*($DRIVER" "$D/simulate-pre.txt" || die "simulation does not install driver $DRIVER"
    grep -q "^Inst dgx-release .*($RELEASE" "$D/simulate-pre.txt" || die "simulation does not install dgx-release $RELEASE"
    r 'sudo -n fwupdmgr get-updates --no-unreported-check 2>&1' > "$D/fwupd-updates-pre.txt" || true   # record only: no capsules on the GX10
    q "PRE DONE: inventory, /etc + selections + grub backup, $(ls "$D/rollback-debs" | wc -l) repacked debs, simulation $(grep -c '^Inst' "$D/simulate-pre.txt") installs"
    touch "$D/pre.done" ;;
  open)
    [ -e "$D/pre.done" ] || die "run pre first"; done_guard
    bc=$(ssh -o BatchMode=yes x86host 'bash ~/dsv41-offload/prod_window.sh check' 2>&1 | tail -2) || true; echo "$bc" >> "$LOG"
    grep -q "PRODUCTION HOLD OK" <<< "$bc" || die "x86host is not in PRODUCTION HOLD OK; fix that before taking production down"
    h 'cd ~/dsv41-deploy && offload/launch/window.sh open "FW2 sparkworker DGX OS 7.6 / kernel 7.0 update, matching sparkhead (coordinator 2026-10-26)"' | tee -a "$LOG" || die "window.sh open refused"
    h 'cd ~/dsv41-deploy && ./stop.sh </dev/null' >> "$LOG" 2>&1 || die "stop.sh failed"
    for n in spark sparkworker; do
      out=$(h "ssh -o BatchMode=yes $n 'docker ps --format {{.Names}} | grep -c dsv41 || true'"); [ "$out" = 0 ] || die "$n still runs a dsv41 container"
    done
    q "OPEN DONE: HOLD written, production stopped on both ranks" ; touch "$D/open.done" ;;
  upgrade)
    [ -e "$D/open.done" ] || die "run open first"; done_guard
    # re-simulate right before installing: archive drift since pre stops here, before anything is touched
    if [ ! -e "$D/apt-upgrade.log" ]; then   # first attempt only (S43 R2-F1): after an install attempt the archive no longer lists these
    r 'sudo -n apt-get update -qq && apt-get -s -o Dpkg::Options::=--force-confold dist-upgrade' > "$D/simulate-upgrade.txt" 2>&1 || die "re-simulation failed"
    grep -q "^Inst linux-image-$KERNEL" "$D/simulate-upgrade.txt" && grep -q "^Inst nvidia-driver-580-open .*($DRIVER" "$D/simulate-upgrade.txt" \
      && grep -q "^Inst dgx-release .*($RELEASE" "$D/simulate-upgrade.txt" || die "re-simulation no longer installs $KERNEL / $DRIVER / $RELEASE; nothing installed, production can restart as is"
    fi
    # detached from this ssh session (S43 F4): a dropped session must not kill dpkg mid-configure
    r 'sudo -n rm -f /var/tmp/fw1-apt.log; sudo -n systemd-run --unit=fw1-apt --collect --wait -p StandardOutput=append:/var/tmp/fw1-apt.log -p StandardError=append:/var/tmp/fw1-apt.log -E DEBIAN_FRONTEND=noninteractive sh -c "apt-get update -qq && apt-get -y -q -o Dpkg::Options::=--force-confold dist-upgrade"' > "$D/apt-upgrade.log" 2>&1 \
      || die "apt dist-upgrade failed or the session dropped: check 'systemctl status fw1-apt' on sparkworker (it may still be running; never reboot or start production while it is); log /var/tmp/fw1-apt.log. NOT rebooted. If dpkg touched the 580 packages production CANNOT restart before a reboot (NVML driver/library mismatch; check: ssh sparkworker-wired nvidia-smi -L): runbook Rollback, before reboot"
    r 'cat /var/tmp/fw1-apt.log' >> "$D/apt-upgrade.log" 2>&1 || true
    r "set -e; test \"\$(dpkg --audit | wc -l)\" = 0; grep -q 'DGX_OTA_VERSION=\"$RELEASE\"' /etc/dgx-release
       for m in nvidia.ko mlx5_core.ko r8127.ko; do find /lib/modules/$KERNEL -name \"\$m*\" | grep -q . || { echo \"missing \$m for $KERNEL\"; exit 1; }; done
       sudo -n grep -q \"with Linux $KERNEL'\" /boot/grub/grub.cfg; sudo -n grep -q \"with Linux $OLD_KERNEL'\" /boot/grub/grub.cfg
       dpkg -s nvidia-driver-580-open | grep -q \"Version: $DRIVER\"" >> "$LOG" 2>&1 || die "pre-reboot gate failed (dpkg audit, release, 7.0 modules nvidia/mlx5/r8127, both GRUB entries, driver $DRIVER)"
    q "UPGRADE DONE: packages installed, gates pass (no firmware capsules on the GX10)"
    touch "$D/upgrade.done" ;;
  reboot)
    [ -e "$D/upgrade.done" ] || die "run upgrade first"; done_guard
    # the CX-7 firmware burn (nvidia-spark-mlnx-firmware-manager, started by its postinst) must finish before the reboot
    wait_end=$((SECONDS + 1200))
    fwm() { r 'systemctl is-active nvidia-spark-mlnx-firmware-manager 2>/dev/null || true'; }   # an ssh failure prints nothing: keep waiting
    until fwm | grep -qxE 'inactive|failed' || [ $SECONDS -ge $wait_end ]; do sleep 20; done
    fwm | grep -qxE 'inactive|failed' || die "CX-7 firmware manager not finished after 20 min (or sparkworker unreachable); do not reboot (read its journal)"
    r 'sudo -n journalctl -b -u nvidia-spark-mlnx-firmware-manager --no-pager 2>&1 | tail -40' > "$D/mlnx-fw-manager.log" 2>&1 || true
    old=$(r cat /proc/sys/kernel/random/boot_id) || die "cannot read boot_id"
    r 'sudo -n systemctl reboot' || q "reboot command returned non-zero (the session may have dropped); waiting for a NEW boot_id"
    q "REBOOT issued; waiting up to 45 min for a new boot (firmware capsules may reboot more than once; never power it off)"
    sleep 60; deadline=$((SECONDS + 2640))
    until b=$(r cat /proc/sys/kernel/random/boot_id 2>/dev/null) && [ -n "$b" ] && [ "$b" != "$old" ] || [ $SECONDS -ge $deadline ]; do sleep 30; done
    b=$(r cat /proc/sys/kernel/random/boot_id 2>/dev/null) && [ "$b" != "$old" ] || die "sparkworker not back on a NEW boot after 45 min (same boot = the reboot never happened): ask NDM to look at it (power LED, display); do NOT power-cycle during a firmware flash"
    r 'tmux has-session -t dsv41-hold 2>/dev/null || tmux new-session -d -s dsv41-hold "exec -a \"vllm serve [DSV41-OFFLOAD hold]\" sleep 86400"'
    h 'test -f ~/dsv41-deploy/HOLD' || die "HOLD file missing on sparkhead after the sparkworker reboot"
    q "REBOOT DONE: sparkworker back on a new boot; hold placeholder re-created on sparkworker" ; touch "$D/reboot.done" ;;
  post)
    [ -e "$D/reboot.done" ] || die "run reboot first"
    inventory post
    [ "$(r uname -r)" = "$KERNEL" ] || die "kernel is $(r uname -r), not $KERNEL"
    r "grep -q 'DGX_OTA_VERSION=\"$RELEASE\"' /etc/dgx-release" || die "dgx-release is not $RELEASE"
    r 'grep -qw kho=off /proc/cmdline' || die "kho=off missing from the kernel command line (nvidia-spark-grub-kho): RDMA memory registration can fail under pressure on 7.0"
    [ "$(r 'nvidia-smi --query-gpu=driver_version --format=csv,noheader')" = "$DRIVER" ] || die "nvidia-smi driver is not $DRIVER"
    for l in $LINKS; do r "ip -br addr show ${l%%=*} | grep -q 'UP .*${l#*=}'" || die "link ${l%%=*} is not UP with ${l#*=}"; done
    for p in $PEERS; do r "ping -c 3 -W 2 $p >/dev/null" || die "no ping to $p"; done
    [ "$(r 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')" -ge 4 ] || die "fewer than 4 RDMA ports PORT_ACTIVE"
    h 'ssh -o BatchMode=yes sparkworker true' || die "sparkhead cannot reach sparkworker (sparkworker) over the CX-7 link"
    IMAGE=$(h 'cd ~/dsv41-deploy && . ./dsv41.env >/dev/null 2>&1; echo $IMAGE')
    r "docker run --rm --pull never --gpus all --entrypoint nvidia-smi $IMAGE -L" > "$D/docker-gpu.txt" 2>&1 || die "docker GPU run failed (read $D/docker-gpu.txt)"
    r 'sudo -n journalctl -k -b --no-pager 2>/dev/null | grep -iE "NVRM: Xid|mlx5.*(error|fail)|r8127.*(error|fail)|BUG:|Oops" | head -20' > "$D/kernel-errors.txt" || true
    # a non-empty log is a stop for a human look; after reading it, FW1_ACCEPT_KLOG=1 records it and goes on
    if [ -s "$D/kernel-errors.txt" ]; then
      [ "${FW1_ACCEPT_KLOG:-0}" = 1 ] || die "kernel log has GPU/NIC error lines (read $D/kernel-errors.txt; rerun post with FW1_ACCEPT_KLOG=1 if benign)"
      q "kernel log lines accepted after review: $(wc -l < "$D/kernel-errors.txt")"
    fi
    q "CX-7 after reboot: $(grep -m1 enp1s0f0np0 "$D/inventory-post.txt")"
    q "POST DONE: kernel $KERNEL, release $RELEASE, driver $DRIVER, 3 CX-7 links + RDMA + sparkhead/x86host reachable, docker GPU OK; $(grep -E 'MemTotal' "$D/inventory-post.txt" | tr -s ' ') (pre: $(grep -E 'MemTotal' "$D/inventory-pre.txt" | tr -s ' '); MemAvailable compares only serving to serving, after restore)"
    touch "$D/post.done" ;;
  restore)
    [ -e "$D/post.done" ] || die "run post first"; done_guard   # a second restore would restart production (S43 R3 note 5)
    bash $C/W8/restore.sh start | tee "$D/restore-start.txt" | tee -a "$LOG"
    grep -q "K1 START VERIFIED" "$D/restore-start.txt" || die "no K1 START VERIFIED (read $D/restore-start.txt)"
    bash $C/W8/restore.sh check | tee "$D/restore-check.txt" | tee -a "$LOG"
    [ "$(grep -cE 'greedy K1-post vs restored: [0-9]+ IDENTICAL [0-9]+$' "$D/restore-check.txt")" = 6 ] || die "not six greedy IDENTICAL lines (read $D/restore-check.txt); production stays in the window"
    grep -q "PRODUCTION HOLD OK" "$D/restore-check.txt" || die "no PRODUCTION HOLD OK"
    bash $C/W8/restore.sh close | tee "$D/restore-close.txt" | tee -a "$LOG"; grep -q "CLOSE: window closed" "$D/restore-close.txt" || die "window did not close"
    meminfo post-serving || q "post-serving meminfo snapshot failed (production already restored and the window closed)"
    q "RESTORE DONE: K1 START VERIFIED, six greedy IDENTICAL, PRODUCTION HOLD OK, window closed (both Sparks on $KERNEL / $DRIVER)"
    touch "$D/restore.done" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
