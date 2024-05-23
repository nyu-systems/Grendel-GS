import struct

def decode_binary_file(infile):
    with open(infile, 'rb') as f:
        data = f.read()
    # Define the format string for struct.unpack
    format_string = '=B f f f H H f f f f f f f f f f f'

    # Unpack the data
    unpacked_data = struct.unpack(format_string, data)

    # Map the unpacked data to the respective variables
    decoded_data = {
        'version': unpacked_data[0],
        'focal': unpacked_data[1],
        'k1': unpacked_data[2],
        'k2': unpacked_data[3],
        'w': unpacked_data[4],
        'h': unpacked_data[5],
        'pos_x': unpacked_data[6],
        'pos_y': unpacked_data[7],
        'pos_z': unpacked_data[8],
        'rot_w': unpacked_data[9],
        'rot_x': unpacked_data[10],
        'rot_y': unpacked_data[11],
        'rot_z': unpacked_data[12],
        'fov': unpacked_data[13],
        'aspect': unpacked_data[14],
        'znear': unpacked_data[15],
        'zfar': unpacked_data[16],
    }
    return decoded_data

file_path = "/global/homes/j/jy-nyu/gaussian-splatting/rubble-cameras/close.bin"
file_path = "/global/homes/j/jy-nyu/gaussian-splatting/rubble-cameras/further.bin"
file_path = "/global/homes/j/jy-nyu/gaussian-splatting/rubble-cameras/far.bin"
file_path = "/global/homes/j/jy-nyu/gaussian-splatting/rubble-cameras/medium-val000249.bin"

print(decode_binary_file(file_path))