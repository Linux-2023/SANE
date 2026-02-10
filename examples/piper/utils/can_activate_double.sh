#!/bin/bash

# The default CAN name can be set by the user via command-line parameters.
DEFAULT_CAN_NAME="${1:-can0}"

# The default bitrate for a single CAN module can be set by the user via command-line parameters.
DEFAULT_BITRATE="${2:-1000000}"

# USB hardware address (optional parameter)
USB_ADDRESS="${3}"
echo "-------------------START-----------------------"
# Check if ethtool is installed.
if ! dpkg -l | grep -q "ethtool"; then
    echo "Error: ethtool not detected in the system."
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# Check if can-utils is installed.
if ! dpkg -l | grep -q "can-utils"; then
    echo "Error: can-utils not detected in the system."
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

# Retrieve the number of CAN modules in the current system.
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Check if any CAN modules are detected.
if [ "$CURRENT_CAN_COUNT" -eq "0" ]; then
    echo "Error: No CAN interface detected."
    echo "-------------------ERROR-----------------------"
    exit 1
fi

if [ -n "$USB_ADDRESS" ]; then
    echo "Detected USB hardware address parameter: $USB_ADDRESS"
    
    # Use ethtool to find the CAN interface corresponding to the USB hardware address.
    INTERFACE_NAME=""
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        if [ "$BUS_INFO" = "$USB_ADDRESS" ]; then
            INTERFACE_NAME="$iface"
            break
        fi
    done
    
    if [ -z "$INTERFACE_NAME" ]; then
        echo "Error: Unable to find CAN interface corresponding to USB hardware address $USB_ADDRESS."
        exit 1
    else
        echo "Found the interface corresponding to USB hardware address $USB_ADDRESS: $INTERFACE_NAME."
        INTERFACES_TO_ACTIVATE="$INTERFACE_NAME"
    fi
else
    # If no USB address is provided, activate all detected CAN interfaces.
    INTERFACES_TO_ACTIVATE=$(ip -br link show type can | awk '{print $1}')
    echo "Detected $CURRENT_CAN_COUNT CAN interface(s): $INTERFACES_TO_ACTIVATE"
    if [ "$CURRENT_CAN_COUNT" -gt "1" ]; then
        echo "Multiple interfaces detected. Activating all with bitrate $DEFAULT_BITRATE."
    fi
fi

# Iterate through all interfaces to be activated.
for INTERFACE_NAME in $INTERFACES_TO_ACTIVATE; do
    echo "--- Configuring interface: $INTERFACE_NAME ---"
    
    # Check if the current interface is already activated.
    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")
    
    # Retrieve the bitrate of the current interface.
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')
    
    if [ "$IS_LINK_UP" = "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already activated with a bitrate of $DEFAULT_BITRATE."
    else
        # If the interface is not activated or the bitrate is different, configure it.
        if [ "$IS_LINK_UP" = "yes" ]; then
            echo "Interface $INTERFACE_NAME is already activated, but current bitrate $CURRENT_BITRATE doesn't match $DEFAULT_BITRATE."
        else
            echo "Interface $INTERFACE_NAME is not activated."
        fi
        
        # Set the interface bitrate and activate it.
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
        sudo ip link set "$INTERFACE_NAME" up
        echo "Interface $INTERFACE_NAME has been set to bitrate $DEFAULT_BITRATE and activated."
    fi

    # Rename logic (only if USB_ADDRESS was provided or there is only one interface).
    if [ -n "$USB_ADDRESS" ] || [ "$CURRENT_CAN_COUNT" -eq "1" ]; then
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Rename interface $INTERFACE_NAME to $DEFAULT_CAN_NAME."
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Renamed to $DEFAULT_CAN_NAME."
        fi
    fi
done

echo "-------------------OVER------------------------"
