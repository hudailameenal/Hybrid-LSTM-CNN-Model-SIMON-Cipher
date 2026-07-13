import os
import struct
import math
import random
from cipher import SimonCipher  # Assuming the SimonCipher class is saved in simon_cipher.py

# Configuration
BLOCK_SIZE = 64  # 64-bit blocks
KEY_SIZE = 128   # Simon 64/128 configuration
CTT_STATIC_KEY = 0x1b1a1918131211100b0a090803020100  # Simon test vector key
PDF_FILES = ['pdfs/1.pdf', 'pdfs/2.pdf', 'pdfs/3.pdf']
OUTPUT_DIR = 'datasets'

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def bytes_to_int(b):
    return int.from_bytes(b, byteorder='big')

def int_to_bytes(n):
    return n.to_bytes(8, byteorder='big')

def generate_ctt_incremental():
    """Generate Dataset-1: Incremental CTT"""
    initial = os.urandom(8)
    fixed_part = initial[:5]
    start_value = bytes_to_int(initial[5:])
    cipher = SimonCipher(CTT_STATIC_KEY, key_size=KEY_SIZE, block_size=BLOCK_SIZE)
    
    with open(f'{OUTPUT_DIR}/Dataset-1.csv', 'w') as f:
        f.write("plaintext,ciphertext\n")
        for i in range(2**15):
            var_part = (start_value + i) & 0xFFFFFF
            plaintext_bytes = fixed_part + struct.pack('>I', var_part)[1:]
            plaintext = bytes_to_int(plaintext_bytes)
            ciphertext = cipher.encrypt(plaintext)
            f.write(f"{plaintext:016x},{ciphertext:016x}\n")

def generate_ctt_decremental():
    """Generate Datasets 2-4: Decremental CTT"""
    for ds_num in range(2, 5):
        initial = os.urandom(8)
        fixed_part = initial[:5]
        start_value = bytes_to_int(initial[5:])
        cipher = SimonCipher(CTT_STATIC_KEY, key_size=KEY_SIZE, block_size=BLOCK_SIZE)
        
        with open(f'{OUTPUT_DIR}/Dataset-{ds_num}.csv', 'w') as f:
            f.write("plaintext,ciphertext\n")
            for i in range(2**11):
                var_part = (start_value - i) & 0xFFFFFF
                plaintext_bytes = fixed_part + struct.pack('>I', var_part)[1:]
                plaintext = bytes_to_int(plaintext_bytes)
                ciphertext = cipher.encrypt(plaintext)
                f.write(f"{plaintext:016x},{ciphertext:016x}\n")

def generate_nctt():
    """Generate Datasets 5-7: Non-correlated from PDFs"""
    sizes = [int(2**16.3), int(2**14.6), int(2**15.3)]
    
    for i, (pdf, size) in enumerate(zip(PDF_FILES, sizes), start=5):
        # Generate unique random key per dataset
        key = random.getrandbits(KEY_SIZE)
        cipher = SimonCipher(key, key_size=KEY_SIZE, block_size=BLOCK_SIZE)
        
        # Read and process PDF
        with open(pdf, 'rb') as f:
            data = f.read()
        
        # Pad data to multiple of 8 bytes
        if len(data) % 8 != 0:
            data += b'\x00' * (8 - len(data) % 8)
        
        # Process chunks
        seen = set()
        count = 0
        with open(f'{OUTPUT_DIR}/Dataset-{i}.csv', 'w') as out:
            out.write("plaintext,ciphertext\n")
            for j in range(0, len(data), 8):
                chunk = data[j:j+8]
                if chunk in seen:
                    continue
                seen.add(chunk)
                
                plaintext = bytes_to_int(chunk)
                ciphertext = cipher.encrypt(plaintext)
                out.write(f"{plaintext:016x},{ciphertext:016x}\n")
                
                count += 1
                if count >= size:
                    break

def main():
    create_dir(OUTPUT_DIR)
    print("Generating Dataset-1 (Incremental CTT)...")
    generate_ctt_incremental()
    
    print("Generating Datasets 2-4 (Decremental CTT)...")
    generate_ctt_decremental()
    
    print("Generating Datasets 5-7 (Non-correlated NCTT)...")
    generate_nctt()
    
    print("All datasets generated successfully!")

if __name__ == "__main__":
    main()