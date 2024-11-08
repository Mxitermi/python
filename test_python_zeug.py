import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Verfügbare Audio-Geräte:")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        
        device_index = device_info["index"]
        device_name = device_info["name"]
        max_input_channels = device_info["maxInputChannels"]
        max_output_channels = device_info["maxOutputChannels"]
        
        # Überprüfen, ob das Gerät Input oder Output unterstützt
        if max_input_channels > 0:
            io_type = "Input"
        elif max_output_channels > 0:
            io_type = "Output"
        else:
            io_type = "Unknown"

        print(f"Index: {device_index}")
        print(f"Name: {device_name}")
        print(f"Typ: {io_type}")
        print(f"Max. Input-Kanäle: {max_input_channels}")
        print(f"Max. Output-Kanäle: {max_output_channels}")
        print("-" * 50)
    
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()
