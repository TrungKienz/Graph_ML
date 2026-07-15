import platform
import socket
import time

print("Hello from remote!")
print("Host:", socket.gethostname())
print("Python:", platform.python_version())

# Tạm dừng chương trình trong 1,000,000 giây
time.sleep(1000000)
