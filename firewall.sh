#!/usr/bin/env bash
# Restrict the API port to trusted networks, and the TP2 coordination port to the fabric (in-memory iptables/ip6tables
# rules; not persistent across reboot; start.sh re-applies them before every start).
#   ./firewall.sh apply | remove | show
set -euo pipefail; cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
ALLOW="127.0.0.0/8 LAN_SUBNET/24 100.64.0.0/10 10.202.26.0/30 10.202.27.0/30 172.16.0.0/12"   # localhost, home LAN, tailscale, fabric, docker
C=DSV41-API
# vLLM's torch.distributed master port (--master-port 25000 in the recipe launcher) listens on every interface, IPv4 and
# IPv6, but only the two ranks use it, over the fabric (worker 10.202.26.1 -> head 10.202.26.2). SPARKFIX1022, 2026-10-22:
# closed to the home LAN (wired and Wi-Fi) and Tailscale.
COORD_PORT=25000
COORD_ALLOW="127.0.0.0/8 10.202.26.0/30 10.202.27.0/30"   # localhost and the fabric only (IPv6: ::1 only)
K=DSV41-COORD
case "${1:-show}" in
  apply)
    sudo -n iptables -N $C 2>/dev/null || sudo -n iptables -F $C
    for net in $ALLOW; do sudo -n iptables -A $C -s $net -j RETURN; done
    sudo -n iptables -A $C -j DROP
    sudo -n iptables -C INPUT -p tcp --dport $PORT -j $C 2>/dev/null || sudo -n iptables -I INPUT 1 -p tcp --dport $PORT -j $C
    sudo -n iptables -N $K 2>/dev/null || sudo -n iptables -F $K
    for net in $COORD_ALLOW; do sudo -n iptables -A $K -s $net -j RETURN; done
    sudo -n iptables -A $K -j DROP
    sudo -n iptables -C INPUT -p tcp --dport $COORD_PORT -j $K 2>/dev/null || sudo -n iptables -I INPUT 1 -p tcp --dport $COORD_PORT -j $K
    sudo -n ip6tables -N $K 2>/dev/null || sudo -n ip6tables -F $K
    sudo -n ip6tables -A $K -s ::1/128 -j RETURN
    sudo -n ip6tables -A $K -j DROP
    sudo -n ip6tables -C INPUT -p tcp --dport $COORD_PORT -j $K 2>/dev/null || sudo -n ip6tables -I INPUT 1 -p tcp --dport $COORD_PORT -j $K
    echo "port $PORT restricted to: $ALLOW"
    echo "port $COORD_PORT restricted to: $COORD_ALLOW ::1" ;;
  remove)
    sudo -n iptables -D INPUT -p tcp --dport $PORT -j $C 2>/dev/null || true
    sudo -n iptables -F $C 2>/dev/null || true; sudo -n iptables -X $C 2>/dev/null || true
    sudo -n iptables -D INPUT -p tcp --dport $COORD_PORT -j $K 2>/dev/null || true
    sudo -n iptables -F $K 2>/dev/null || true; sudo -n iptables -X $K 2>/dev/null || true
    sudo -n ip6tables -D INPUT -p tcp --dport $COORD_PORT -j $K 2>/dev/null || true
    sudo -n ip6tables -F $K 2>/dev/null || true; sudo -n ip6tables -X $K 2>/dev/null || true; echo "removed" ;;
  show)
    sudo -n iptables -S $C 2>/dev/null; sudo -n iptables -S INPUT | grep -- "--dport $PORT" || echo "no rule on port $PORT"
    sudo -n iptables -S $K 2>/dev/null; sudo -n iptables -S INPUT | grep -- "--dport $COORD_PORT" || echo "no IPv4 rule on port $COORD_PORT"
    sudo -n ip6tables -S $K 2>/dev/null; sudo -n ip6tables -S INPUT | grep -- "--dport $COORD_PORT" || echo "no IPv6 rule on port $COORD_PORT" ;;
esac
