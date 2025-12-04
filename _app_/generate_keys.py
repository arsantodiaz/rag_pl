import bcrypt
import sys

# Skrip ini menggunakan bcrypt secara langsung untuk menghindari masalah versi Hasher

try:
    import bcrypt
except ImportError:
    print("Pustaka bcrypt tidak ditemukan. Harap install dengan 'pip install bcrypt'")
    sys.exit(1)

# Ganti '123' dengan kata sandi yang Anda inginkan
password_to_hash = '123'

# Mengubah password menjadi bytes
password_bytes = password_to_hash.encode('utf-8')

# Membuat salt dan hash
hashed_password_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

# Mengubah hash kembali menjadi string untuk ditampilkan dan disalin
hashed_password_str = hashed_password_bytes.decode('utf-8')

print("--- HASH GENERATOR (via Bcrypt) ---")
print(f"Password: '{password_to_hash}'")
print(f"Hashed  : {hashed_password_str}")
print("\nSalin hash di atas dan tempel ke dalam file app_ui.py")
print("-------------------------------------")
