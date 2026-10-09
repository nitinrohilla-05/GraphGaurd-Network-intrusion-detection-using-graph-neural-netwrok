#!/usr/bin/env bash
# GraphGuard Phase P3a - Environment Pre-Flight Verification Script
# Verifies Open vSwitch, OpenFlow 1.3 meters (kernel and userspace datapaths),
# networking tools, and Python dependencies per Decision 56 and Decision 73.

set -euo pipefail

echo "======================================================================"
echo "          GraphGuard Phase P3a - Environment Pre-Flight Check          "
echo "======================================================================"
echo "Date: $(date -u)"
echo "Host: $(hostname)"
echo "Kernel: $(uname -s -r -m)"
echo "======================================================================"

KERNEL_METER_OK="FAIL"
USERSPACE_METER_OK="FAIL"

# 1. Root / Sudo check
if [ "$EUID" -ne 0 ]; then
    echo "[!] Not running as root. Testing sudo access..."
    if sudo -n true 2>/dev/null; then
        SUDO="sudo"
        echo "[+] Sudo access verified."
    else
        echo "[-] Root or passwordless sudo required for Open vSwitch & Mininet operations."
        SUDO="sudo"
    fi
else
    SUDO=""
    echo "[+] Running as root."
fi

# 2. Check for required system binaries
echo ""
echo "--- Checking Core Utilities ---"
REQUIRED_BINS=("ovs-vsctl" "ovs-ofctl" "mn" "python3" "hping3")
MISSING_BINS=()

for b in "${REQUIRED_BINS[@]}"; do
    if command -v "$b" &>/dev/null; then
        echo "[+] $b: $(command -v "$b")"
    else
        echo "[-] $b: NOT FOUND"
        MISSING_BINS+=("$b")
    fi
done

# 3. Start Open vSwitch Service if installed
echo ""
echo "--- Starting Open vSwitch Service ---"
if command -v ovs-vsctl &>/dev/null; then
    if systemctl is-active --quiet openvswitch-switch 2>/dev/null; then
        echo "[+] openvswitch-switch service is active (systemd)."
    elif $SUDO service openvswitch-switch status &>/dev/null 2>&1; then
        echo "[+] openvswitch-switch service is active (sysvinit)."
    else
        echo "[*] Starting openvswitch-switch service..."
        if command -v systemctl &>/dev/null && systemctl list-unit-files openvswitch-switch.service &>/dev/null; then
            $SUDO systemctl start openvswitch-switch || true
        elif [ -x /etc/init.d/openvswitch-switch ]; then
            $SUDO /etc/init.d/openvswitch-switch start || true
        elif command -v service &>/dev/null; then
            $SUDO service openvswitch-switch start || true
        fi
    fi

    # Verify ovs-vswitchd is running
    if pgrep -x ovs-vswitchd &>/dev/null; then
        echo "[+] ovs-vswitchd process is running."
    else
        echo "[-] ovs-vswitchd is not running."
    fi
else
    echo "[-] Open vSwitch not installed."
fi

# 4. OpenFlow 1.3 Meter Test: Kernel Datapath Branch (Decision 73)
echo ""
echo "--- Branch 1: OpenFlow 1.3 Meter Test (Kernel Datapath) ---"
if command -v ovs-vsctl &>/dev/null && command -v ovs-ofctl &>/dev/null; then
    BR_KERNEL="gg-test-kern"
    $SUDO ovs-vsctl --if-exists del-br "$BR_KERNEL" 2>/dev/null || true
    
    if $SUDO ovs-vsctl add-br "$BR_KERNEL" 2>/dev/null; then
        $SUDO ovs-vsctl set bridge "$BR_KERNEL" protocols=OpenFlow13 2>/dev/null || true
        
        # Attempt to add OpenFlow 1.3 rate-limit meter with OFPMBT_DROP band
        if $SUDO ovs-ofctl -O OpenFlow13 add-meter "$BR_KERNEL" "meter=1,pkt_ps,band=type=drop,rate=100" 2>/dev/null; then
            METER_DUMP=$($SUDO ovs-ofctl -O OpenFlow13 dump-meters "$BR_KERNEL" 2>/dev/null || true)
            if echo "$METER_DUMP" | grep -q "meter=1"; then
                echo "[+] Kernel Datapath OF1.3 Meter: SUCCESS"
                echo "    Dump: $METER_DUMP"
                KERNEL_METER_OK="PASS"
            else
                echo "[-] Kernel Datapath OF1.3 Meter: Meter created but dump failed."
            fi
        else
            echo "[-] Kernel Datapath OF1.3 Meter: add-meter failed (kernel module lacks meter support or not loaded)."
        fi
        $SUDO ovs-vsctl del-br "$BR_KERNEL" 2>/dev/null || true
    else
        echo "[-] Failed to create kernel test bridge $BR_KERNEL"
    fi
else
    echo "[-] Skipped: OVS utilities not present."
fi

# 5. OpenFlow 1.3 Meter Test: Userspace Datapath Branch (Decision 73)
echo ""
echo "--- Branch 2: OpenFlow 1.3 Meter Test (Userspace netdev Datapath) ---"
if command -v ovs-vsctl &>/dev/null && command -v ovs-ofctl &>/dev/null; then
    BR_USER="gg-test-user"
    $SUDO ovs-vsctl --if-exists del-br "$BR_USER" 2>/dev/null || true
    
    if $SUDO ovs-vsctl add-br "$BR_USER" -- set bridge "$BR_USER" datapath_type=netdev protocols=OpenFlow13 2>/dev/null; then
        # Attempt to add OpenFlow 1.3 rate-limit meter on userspace bridge
        if $SUDO ovs-ofctl -O OpenFlow13 add-meter "$BR_USER" "meter=1,pkt_ps,band=type=drop,rate=100" 2>/dev/null; then
            METER_DUMP_USER=$($SUDO ovs-ofctl -O OpenFlow13 dump-meters "$BR_USER" 2>/dev/null || true)
            if echo "$METER_DUMP_USER" | grep -q "meter=1"; then
                echo "[+] Userspace Datapath OF1.3 Meter: SUCCESS"
                echo "    Dump: $METER_DUMP_USER"
                USERSPACE_METER_OK="PASS"
            else
                echo "[-] Userspace Datapath OF1.3 Meter: Meter created but dump failed."
            fi
        else
            echo "[-] Userspace Datapath OF1.3 Meter: add-meter failed."
        fi
        $SUDO ovs-vsctl del-br "$BR_USER" 2>/dev/null || true
    else
        echo "[-] Failed to create userspace test bridge $BR_USER"
    fi
else
    echo "[-] Skipped: OVS utilities not present."
fi

# 6. Python Environment and Libraries Check
echo ""
echo "--- Checking Python Environment & P3a Dependencies ---"
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    fi
fi

PY_BIN="python3"
if [ -n "${VIRTUAL_ENV:-}" ] && command -v python &>/dev/null; then
    PY_BIN="python"
elif ! command -v python3 &>/dev/null && command -v python &>/dev/null; then
    PY_BIN="python"
fi

if command -v "$PY_BIN" &>/dev/null; then
    echo "[+] Python binary: $(command -v "$PY_BIN")"
    echo "[+] Python version: $("$PY_BIN" --version)"
    "$PY_BIN" -c "
import sys
mods = ['nfstream', 'scapy', 'paramiko', 'dns', 'requests', 'pymysql']
for m in mods:
    try:
        __import__(m)
        print(f'[+] Python module {m}: OK')
    except Exception as e:
        print(f'[-] Python module {m}: FAILED ({e})')
" || true
else
    echo "[-] python/python3 not found."
fi

# 7. Summary & Recommendation
echo ""
echo "======================================================================"
echo "                     PRE-FLIGHT VERIFICATION SUMMARY                  "
echo "======================================================================"
echo "Kernel Datapath OF1.3 Meter:    $KERNEL_METER_OK"
echo "Userspace Datapath OF1.3 Meter: $USERSPACE_METER_OK"

if [ "$KERNEL_METER_OK" = "PASS" ]; then
    echo "Recommended Mininet Datapath:  kernel (OVSSwitch default)"
elif [ "$USERSPACE_METER_OK" = "PASS" ]; then
    echo "Recommended Mininet Datapath:  userspace (OVSSwitch datapath='user')"
else
    echo "Recommended Mininet Datapath:  NONE (OVS meters unsupported or OVS not installed)"
fi
echo "======================================================================"
