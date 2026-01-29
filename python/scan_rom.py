import os

def scan_rom():
    rom_path = "BattleCity.nes"
    if not os.path.exists(rom_path):
        print("ROM not found!")
        return

    with open(rom_path, "rb") as f:
        data = f.read()

    print(f"File size: {len(data)} bytes")
    
    # Header is 16 bytes
    # PRG is likely 16KB (16384 bytes)
    # CHR is likely 8KB (8192 bytes)
    # 16 + 16384 + 8192 = 24592. Matches file size.
    
    # CPU $8000 maps to PRG offset 0.
    # CPU $EB30 maps to where?
    # Since PRG is 16KB, it is mirrored at $8000 and $C000.
    # $8000 -> 0 (in PRG)
    # $C000 -> 0 (in PRG)
    
    # $EB30 is in the higher mirror ($C000-$FFFF).
    # Offset = $EB30 - $C000 = 0x2B30
    
    # Dump window at E6B0
    start_cpu = 0xE6A0
    end_cpu = 0xE6C0
    
    start_offset = 16 + (start_cpu - 0xC000)
    end_offset = 16 + (end_cpu - 0xC000)
    
    print(f"Dumping bytes from CPU ${start_cpu:04X} to ${end_cpu:04X}:")
    
    for i in range(start_offset, end_offset):
        cpu_addr = 0xC000 + (i - 16)
        byte = data[i]
        print(f"${cpu_addr:04X}: {byte:02X}")

if __name__ == "__main__":
    scan_rom()
