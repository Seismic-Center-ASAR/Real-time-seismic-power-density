from obspy import UTCDateTime
from obspy.clients.seedlink import Client
from scipy.signal import iirfilter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

# Set up the Seedlink client
client = Client("geofon.gfz-potsdam.de", 18000)

# Set the start and end times for the plot
endtime = UTCDateTime.now()  # now
starttime = endtime - 60  # 60 seconds ago

# Set the size of the buffer (in seconds)
buffer_size = 10

# Initialize an empty buffer
buffer = []

# Create an empty plot to use for updating
fig, ax = plt.subplots()
line, = ax.plot([], color='black')

# Define the filter parameters for the frequency analysis
nyquist_freq = 100
low_freq = 1
high_freq = 10
order = 4
b, a = iirfilter(order, [low_freq / nyquist_freq, high_freq / nyquist_freq], btype='band')

# Continuously update the plot with new data
while True:
    # If the buffer is empty or the endtime is greater than the last buffered time, get new data
    if len(buffer) == 0 or endtime > buffer[-1].stats.endtime:
        # Get the waveform data
        st = client.get_waveforms("RO", "VRI", "", "EHZ", starttime, endtime)

        # Append the new data to the buffer
        buffer.append(st[0])
        
        # If the buffer has grown larger than the buffer size, remove the oldest data
        if len(buffer) > buffer_size:
            buffer.pop(0)

    # Combine the data from the buffer into a single trace
    trace = buffer[0]
    for i in range(1, len(buffer)):
        trace += buffer[i]

    # Filter the trace to get the band-limited signal
    data = trace.data
    data_filt = filtfilt(b, a, data)

    # Compute the spectral power density of the signal
    f, psd = plt.psd(data_filt, NFFT=1024, Fs=trace.stats.sampling_rate, visible=False)

    # Find the dominant frequency in the power spectrum
    max_psd_idx = np.argmax(psd)
    dominant_freq = f[max_psd_idx]

    # Update the plot with the new data
    line.set_data(trace.times(), trace.data)
    ax.relim()
    ax.autoscale_view()

    # Check if the amplitude of the signal exceeds the threshold for the alert
    if np.max(np.abs(data)) > 6000:
        # Play the sound alert
        filename = 'alert.wav'
        data, fs = sf.read(filename, dtype='float32')
        sd.play(data, fs)

    # Pause for a short time before getting new data
    plt.pause(0.1)

    # Update the start and end times for the next iteration
    endtime = UTCDateTime.now()
    starttime = endtime - 60  # 60 seconds ago
    
    # Print the dominant frequency of the signal
    print("Dominant frequency: {:.2f} Hz".format(dominant_freq))

