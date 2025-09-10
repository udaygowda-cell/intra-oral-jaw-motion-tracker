import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import deque


port = 'COM3'  
baudrate = 115200
ser = serial.Serial(port, baudrate, timeout=1)
time.sleep(2)

# Set up interactive 3D plot
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Buffers to store recent data points
x_vals = deque(maxlen=500)
y_vals = deque(maxlen=500)
z_vals = deque(maxlen=500)

# Set axis ranges and labels
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_zlim(-2000, 2000)
ax.set_xlabel('Mag X (µT)', color='r')
ax.set_ylabel('Mag Y (µT)', color='r')
ax.set_zlabel('Mag Z (µT)', color='r')

plot_line = None

# Close plot properly
def handle_close(evt):
    print("Plot window closed")
    ser.close()
    exit()

fig.canvas.mpl_connect('close_event', handle_close)

# Main loop
while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            parts = line.split(',')
            if len(parts) == 3:
                x, y, z = map(float, parts)

                # Print values to terminal
                print(f"X: {x:.2f} µT, Y: {y:.2f} µT, Z: {z:.2f} µT")

                # Update deques
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

                # Update plot
                if plot_line:
                    plot_line.remove()
                plot_line = ax.plot3D(list(x_vals), list(y_vals), list(z_vals), color='blue')[0]

                plt.draw()
                plt.pause(0.05)
        else:
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("Stopped by user")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

ser.close()
