#!/usr/bin/env bash
# Restrict the API port to trusted networks (in-memory iptables rules; not persistent across reboot; start.sh re-applies).
#   ./firewall.sh apply | remove | show
set -euo pipefail; cd "$(dirname "$0")"; set -a; . ./dsv41.env; set +a
ALLOW="127.0.0.0/8 LAN_SUBNET/24 100.64.0.0/10 10.202.26.0/30 10.202.27.0/30 172.16.0.0/12"   # localhost, home LAN, tailscale, fabric, docker
C=DSV41-API
case "${1:-show}" in
  apply)
    sudo -n iptables -N $C 2>/dev/null || sudo -n iptables -F $C
    for net in $ALLOW; do sudo -n iptables -A $C -s $net -j RETURN; done
    sudo -n iptables -A $C -j DROP
    sudo -n iptables -C INPUT -p tcp --dport $PORT -j $C 2>/dev/null || sudo -n iptables -I INPUT 1 -p tcp --dport $PORT -j $C
    echo "port $PORT restricted to: $ALLOW" ;;
  remove)
    sudo -n iptables -D INPUT -p tcp --dport $PORT -j $C 2>/dev/null || true
    sudo -n iptables -F $C 2>/dev/null || true; sudo -n iptables -X $C 2>/dev/null || true; echo "removed" ;;
  show) sudo -n iptables -S $C 2>/dev/null; sudo -n iptables -S INPUT | grep -- "--dport $PORT" || echo "no rule on port $PORT" ;;
esac
