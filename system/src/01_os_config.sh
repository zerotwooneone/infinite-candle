#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Check for Root
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run as root (sudo bash 01_os_config.sh)"
  exit
fi

echo "🔄 Updating Package Lists..."
apt-get update && apt-get upgrade -y

echo "🔇 Disabling Onboard Audio (to fix PWM timing)..."

# Detect OS version for config.txt location
# Bookworm (Debian 12) uses /boot/firmware/config.txt
# Bullseye (Debian 11) uses /boot/config.txt
if [ -f /boot/firmware/config.txt ]; then
    CONFIG_FILE="/boot/firmware/config.txt"
else
    CONFIG_FILE="/boot/config.txt"
fi

echo "   -> Detected config file at: $CONFIG_FILE"

# Backup the config file before editing
cp "$CONFIG_FILE" "$CONFIG_FILE.bak"

# Disable audio parameter
# sed command searches for 'dtparam=audio=on' and replaces it with 'off'
if grep -q "dtparam=audio=on" "$CONFIG_FILE"; then
    sed -i 's/dtparam=audio=on/dtparam=audio=off/g' "$CONFIG_FILE"
    echo "   -> Audio parameter disabled in config.txt."
else
    echo "   -> Audio parameter already off or not found."
fi

# Create a blacklist file to prevent sound modules from loading
echo "🚫 Blacklisting sound kernel modules..."
cat > /etc/modprobe.d/snd-blacklist.conf <<EOF
blacklist snd_bcm2835
blacklist snd_pcm
blacklist snd_timer
blacklist snd
EOF

echo "✅ OS Configuration Complete."
echo "⚠️  PLEASE REBOOT YOUR PI NOW (sudo reboot)"