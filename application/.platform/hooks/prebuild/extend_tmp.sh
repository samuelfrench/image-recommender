#!/bin/bash
echo "Setting up 4GB tmpfs for /tmp"

# Unmount existing /tmp if mounted
if mountpoint -q /tmp; then
    sudo umount /tmp
fi

# Remount /tmp with tmpfs and 4GB size
sudo mount -t tmpfs -o size=4G tmpfs /tmp

# Add to /etc/fstab to persist across reboots
if ! grep -q "tmpfs /tmp tmpfs defaults,size=4G 0 0" /etc/fstab; then
    sudo su
    echo "tmpfs /tmp tmpfs defaults,size=4G 0 0" >> /etc/fstab
    exit
fi