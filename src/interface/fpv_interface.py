import logging
import serial
import struct
import threading
import time
from typing import List, Callable, Optional
from serial.serialutil import SerialException
import signal
import rclpy
from fpv_node import FpvNode

runtime_exec = True

class FpvInterface(threading.Thread):
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUD_RATE = 921000
    PACKET_RATE_SEC = 1/1000.0
    
    SYNC_BYTE = 0xC8
    BROADCAST_ADDR = 0x00
    HANDSET_ADDR = 0xEA
    TXMODULE_ADDR = 0xEE
   
    CHANNELS_FRAME = 0x16
    PING_DEVICES_FRAME = 0x28
    DEVICE_INFO_FRAME = 0x29
    RADIO_FRAME = 0x3A
    LINK_STATS_FRAME = 0x14
    BATTERY_SENSOR_FRAME = 0x08
    CRSF_FRAMETYPE_IMU = 0x2E
    CRSF_FRAMETYPE_GPS = 0x02
    CRSF_FRAMETYPE_ATTITUDE = 0x1E
    CRSF_FRAMETYPE_FLIGHTMODE = 0x21

    CHANNEL_MIN_1000 = 191
    CHANNEL_MID_1500 = 992
    CHANNEL_MAX_2000 = 1792
    
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.buffer = bytearray()
        self.running = False
        self.channels = [500] * 16
        
        self.device_info = []
        self.radio_id_data = []
        self.link_status_data = []

        self.battery_data = []
        self.attitude_data = []
        self.imu_data = []

        self.device_info_callback: Optional[Callable[[tuple], None]] = None
        self.radio_id_callback: Optional[Callable[[tuple], None]] = None
        self.link_status_callback: Optional[Callable[[tuple], None]] = None
        self.battery_callback: Optional[Callable[[tuple], None]] = None
        self.attitude_callback: Optional[Callable[[tuple], None]] = None
        self.imu_callback: Optional[Callable[[tuple], None]] = None
       

        rclpy.init()
        self.ros2_node = FpvNode()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler("flight_control_interface.log")
        console_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.serial_conn = None
        self.last_ping = time.time()

    def run(self):
        self.running = True
        self.logger.info("CRSF TX comms started")
        last_time = time.time()
        try:
            with serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=0) as ser:
                self.serial_conn = ser
                self.logger.info(f"connected to {self.SERIAL_PORT} @ {self.BAUD_RATE} baud")
                while self.running:
                    if time.time() > last_time + 1/500.0:
                        last_time = time.time()
                        self.send_channel_data()
                    self.read_incoming_data()
                    time.sleep(self.PACKET_RATE_SEC)
        except SerialException as e:
            self.logger.error(f"SerialException: {e}")
        finally:
            self.running = False
    
    def stop(self):
        self.running = False

    def set_device_info_callback(self, callback: Callable[[tuple], None]):
        self.device_info_callback = callback

    def set_radio_id_callback(self, callback: Callable[[tuple], None]):
        self.radio_id_callback = callback

    def set_link_status_callback(self, callback: Callable[[tuple], None]):
        self.link_status_callback = callback

    def set_battery_callback(self, callback: Callable[[tuple], None]):
        self.battery_callback = callback

    def set_attitude_callback(self, callback: Callable[[tuple], None]):
        self.attitude_callback = callback

    def set_imu_callback(self, callback: Callable[[tuple], None]):
        self.imu_callback = callback

    def send_channel_data(self):
        packet = self.create_packet_channels(self.channels)
        self.serial_conn.write(packet)

    def read_incoming_data(self):
        try:
            incoming_data = self.serial_conn.read(64)
            self.buffer.extend(incoming_data)
            self.buffer = self.process_packet(self.buffer)
        except SerialException:
            self.logger.error("error reading data from serial")

    def scale_channel_value(self, input_value):
        """Scale the input value from the range 1000-2000 to CRSF 191-1792."""
        return int(self.CHANNEL_MIN_1000 + (input_value - 1000) * (self.CHANNEL_MAX_2000 - self.CHANNEL_MIN_1000) / 1000)

    def create_packet_channels(self, channels: List[int]) -> bytes:
        crossfire_ch_bits = 11
        payload_size = 22
        buf = bytearray(26)
        bit_offset = 0

        # Set header information
        buf[0] = self.TXMODULE_ADDR
        buf[1] = payload_size + 2
        buf[2] = self.CHANNELS_FRAME

        read_value = 0
        bits_merged = 0

        for _, val in enumerate(channels):
            channel_bits = self.scale_channel_value(val) & ((1 << crossfire_ch_bits) - 1)
            read_value |= (channel_bits << bits_merged)
            bits_merged += crossfire_ch_bits

            # Write the accumulated bits into the buffer
            while bits_merged >= 8:
                buf[3 + bit_offset] = read_value & 0xFF
                read_value >>= 8
                bits_merged -= 8
                bit_offset += 1

        # Write remaining bits, if any
        if bits_merged > 0:
            buf[3 + bit_offset] = read_value & 0xFF

        buf[25] = self.crsf_crc8(buf[2:25])
        #self.logger.info(f"Raw Packet (Hex): {' '.join(f'{b:02X}' for b in buf)}")

        return bytes(buf)

    def create_packet_ping_device(self) -> bytes:
        frame = [
            self.SYNC_BYTE, 4, self.PING_DEVICES_FRAME,
            self.BROADCAST_ADDR, self.HANDSET_ADDR, 0
        ]
        frame[5] = self.crsf_crc8(frame[2:5])
        return bytes(frame)

    def crsf_crc8(self, data):
        crc_table = [
            0x00, 0xD5, 0x7F, 0xAA, 0xFE, 0x2B, 0x81, 0x54, 0x29, 0xFC, 0x56, 0x83, 0xD7, 0x02, 0xA8, 0x7D,
            0x52, 0x87, 0x2D, 0xF8, 0xAC, 0x79, 0xD3, 0x06, 0x7B, 0xAE, 0x04, 0xD1, 0x85, 0x50, 0xFA, 0x2F,
            0xA4, 0x71, 0xDB, 0x0E, 0x5A, 0x8F, 0x25, 0xF0, 0x8D, 0x58, 0xF2, 0x27, 0x73, 0xA6, 0x0C, 0xD9,
            0xF6, 0x23, 0x89, 0x5C, 0x08, 0xDD, 0x77, 0xA2, 0xDF, 0x0A, 0xA0, 0x75, 0x21, 0xF4, 0x5E, 0x8B,
            0x9D, 0x48, 0xE2, 0x37, 0x63, 0xB6, 0x1C, 0xC9, 0xB4, 0x61, 0xCB, 0x1E, 0x4A, 0x9F, 0x35, 0xE0,
            0xCF, 0x1A, 0xB0, 0x65, 0x31, 0xE4, 0x4E, 0x9B, 0xE6, 0x33, 0x99, 0x4C, 0x18, 0xCD, 0x67, 0xB2,
            0x39, 0xEC, 0x46, 0x93, 0xC7, 0x12, 0xB8, 0x6D, 0x10, 0xC5, 0x6F, 0xBA, 0xEE, 0x3B, 0x91, 0x44,
            0x6B, 0xBE, 0x14, 0xC1, 0x95, 0x40, 0xEA, 0x3F, 0x42, 0x97, 0x3D, 0xE8, 0xBC, 0x69, 0xC3, 0x16,
            0xEF, 0x3A, 0x90, 0x45, 0x11, 0xC4, 0x6E, 0xBB, 0xC6, 0x13, 0xB9, 0x6C, 0x38, 0xED, 0x47, 0x92,
            0xBD, 0x68, 0xC2, 0x17, 0x43, 0x96, 0x3C, 0xE9, 0x94, 0x41, 0xEB, 0x3E, 0x6A, 0xBF, 0x15, 0xC0,
            0x4B, 0x9E, 0x34, 0xE1, 0xB5, 0x60, 0xCA, 0x1F, 0x62, 0xB7, 0x1D, 0xC8, 0x9C, 0x49, 0xE3, 0x36,
            0x19, 0xCC, 0x66, 0xB3, 0xE7, 0x32, 0x98, 0x4D, 0x30, 0xE5, 0x4F, 0x9A, 0xCE, 0x1B, 0xB1, 0x64,
            0x72, 0xA7, 0x0D, 0xD8, 0x8C, 0x59, 0xF3, 0x26, 0x5B, 0x8E, 0x24, 0xF1, 0xA5, 0x70, 0xDA, 0x0F,
            0x20, 0xF5, 0x5F, 0x8A, 0xDE, 0x0B, 0xA1, 0x74, 0x09, 0xDC, 0x76, 0xA3, 0xF7, 0x22, 0x88, 0x5D,
            0xD6, 0x03, 0xA9, 0x7C, 0x28, 0xFD, 0x57, 0x82, 0xFF, 0x2A, 0x80, 0x55, 0x01, 0xD4, 0x7E, 0xAB,
            0x84, 0x51, 0xFB, 0x2E, 0x7A, 0xAF, 0x05, 0xD0, 0xAD, 0x78, 0xD2, 0x07, 0x53, 0x86, 0x2C, 0xF9
        ]
        crc = 0
        for byte in data:
            crc = crc_table[crc ^ byte]
        return crc

    def process_packet(self, buffer):
        while len(buffer) > 0:
            if buffer[0] != self.HANDSET_ADDR:
                self.logger.error(f"Unexpected packet address: {buffer[0]:02X}")
                buffer.pop(0)
                continue

            packet_len = buffer[1]
            total_packet_size = packet_len + 2

            if len(buffer) < total_packet_size:
                self.logger.debug("total packets size small")
                break

            packet = buffer[:total_packet_size]
            buffer = buffer[total_packet_size:]

            if not self.is_valid_packet(packet):
                self.logger.error(f"Invalid packet: type: {packet[2]} length:{packet[1]}")
                break
           
            type_byte = packet[2]
            payload = packet[3:-1]  # Payload is the data excluding header and CRC
            
            if type_byte == self.DEVICE_INFO_FRAME:
                self.parse_device_info_frame(payload)
                self.logger.debug(f"device info packet: {self.device_info}")
            elif type_byte == self.RADIO_FRAME:
                self.parse_radio_id_frame(payload)
                self.logger.debug(f"radio id packet: {self.radio_id_data}")
            elif type_byte == self.LINK_STATS_FRAME:
                self.parse_link_status(payload)
                self.logger.debug(f"link status packet: {self.link_status_data}")
            elif type_byte == self.BATTERY_SENSOR_FRAME:
                self.parse_battery_frame(payload)
                self.ros2_node.publish_battery(self.battery_data)
                self.logger.debug(f"battery packet: {self.battery_data}")
            elif type_byte == self.CRSF_FRAMETYPE_GPS:
                self.logger.debug(f"gps packet: {payload.hex()}")
            elif type_byte == self.CRSF_FRAMETYPE_ATTITUDE:
                self.parse_attitude_frame(payload)
                self.ros2_node.publish_attitude(self.attitude_data)
                self.logger.debug(f"attitude packet: {self.attitude_data}")
            elif type_byte == self.CRSF_FRAMETYPE_FLIGHTMODE:
                self.logger.debug(f"flight mode: {payload.hex()}")
            elif type_byte == self.CRSF_FRAMETYPE_IMU:
                self.parse_imu_frame(payload)
                self.ros2_node.publish_imu(self.imu_data)
                self.logger.debug(f"imu packet: {self.imu_data}")
            else:
                self.logger.error(f"Unknown Type: {type_byte:#02x}")
                self.logger.error(f"Payload: {payload.hex()}")
        return buffer

    def is_valid_packet(self, packet):
        packet_len = packet[1]
        crc_start = 2
        crc_end = 2 + packet_len - 1
        expected_crc = self.crsf_crc8(packet[crc_start:crc_end])
        return packet[-1] == expected_crc

    def set_channel_values(self, chT, chR, chE, chA, aux1=2000, aux2=2000, aux3=2000, aux4=2000, aux5=2000, aux6=2000, aux7=2000, aux8=2000, aux9=2000, aux10=2000, aux11=2000, aux12=2000):
        self.channels[:16] = [chT, chR, chE, chA, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12]
        
    def set_specific_channel(self, index, value):
        if 0 <= index < 16:
            self.channels[index] = value

    def parse_device_info_frame(self, payload):
        display_name_start = 3
        display_name_end = payload.find(b'\x00')
        display_name = payload[display_name_start:display_name_end].decode('utf-8')
        remaining_payload = payload[display_name_end + 1:]

        serial_number, hw_version, sw_version, config_param_count, config_param_protocol_version = struct.unpack(
            '>III2B', remaining_payload[:14])
        self.device_info = display_name, serial_number, hw_version, sw_version, config_param_count, config_param_protocol_version
        if self.device_info_callback:
            self.device_info_callback(self.device_info)

    def parse_radio_id_frame(self, payload):
        payload = payload[3:]  # Ignore [DEST] [SRC]
        packet_interval, phase_shift = struct.unpack('>ii', payload[:8])
        packet_interval_us = packet_interval / 10
        packet_frequency_hz = 1 / (packet_interval_us / 1_000_000)

        self.radio_id_data = packet_interval_us, packet_frequency_hz, phase_shift
        if self.radio_id_callback:
            self.radio_id_callback(self.radio_id_data)


    def parse_link_status(self, payload):
        uplink_rssi_ant1, uplink_rssi_ant2, uplink_lq, uplink_snr, diversity_antenna, rf_mode, uplink_tx_power, downlink_rssi, downlink_lq, downlink_snr = struct.unpack(
            '>3B2b5B', payload[:10])
        self.link_status_data = -uplink_rssi_ant1,  -uplink_rssi_ant2, uplink_lq, uplink_snr / 4, diversity_antenna, rf_mode, uplink_tx_power, -downlink_rssi, downlink_lq, downlink_snr / 4
        if self.link_status_callback:
            self.link_status_callback(self.link_status_data)

    def parse_battery_frame(self, payload):
        voltage_mv, current_10ma = struct.unpack('>HH', payload[:4])
        mAh_high, mAh_mid, mAh_low = struct.unpack('BBB', payload[4:7])
        mAh_drawn = (mAh_high << 16) | (mAh_mid << 8) | mAh_low
        battery_percentage = struct.unpack('B', payload[7:8])[0]
        voltage_v = voltage_mv / 1000
        current_a = current_10ma / 10
        self.battery_data = voltage_v,current_a,mAh_drawn,battery_percentage
        if self.battery_callback:
            self.battery_callback(self.battery_data)

    def parse_attitude_frame(self, payload):
        try:
            pitch_raw, roll_raw, yaw_raw = struct.unpack('>HHH', payload[:6])

            # unsigned -> signed
            pitch_signed = pitch_raw - 32768
            roll_signed = roll_raw - 32768
            yaw_signed = yaw_raw - 32768

            # milli-radians to radians
            pitch_radians = pitch_signed / 1000.0
            roll_radians = roll_signed / 1000.0
            yaw_radians = yaw_signed / 1000.0

            # Store parsed data
            self.attitude_data = pitch_radians, roll_radians, yaw_radians

            if self.attitude_callback:
                self.attitude_callback(self.attitude_data)

        except Exception as e:
            self.logger.error(f"Failed to parse attitude: {e}")


    def parse_imu_frame(self, payload):
        try:
            acc_x_int, acc_y_int, acc_z_int, vel_x_int, vel_y_int, vel_z_int, time_stamp_ms = struct.unpack('>HHHHHHI', payload[:16])

            # milli-g -> g
            acc_x = (acc_x_int - 32768) / 1000.0
            acc_y = (acc_y_int - 32768) / 1000.0
            acc_z = (acc_z_int - 32768) / 1000.0

            # milli-radians to radians
            vel_x = (vel_x_int - 32768) / 1000.0
            vel_y = (vel_y_int - 32768) / 1000.0
            vel_z = (vel_z_int - 32768) / 1000.0

            self.imu_data = acc_x,acc_y,acc_z,vel_x,vel_y,vel_z,time_stamp_ms

            if self.imu_callback:
                self.imu_callback(self.imu_data)

        except Exception as e:
            self.logger.error(f"Failed to parse imu packet: {e}")


    def get_device_info(self):
        return self.device_info

    def get_radio_id_data(self):
        return self.radio_id_data

    def get_link_status(self):
        return self.link_status_data

def signal_handler(sig, frame):
    global runtime_exec
    print("\nSIGINT received. Exiting gracefully...")
    runtime_exec = False

if __name__ == "__main__":
    def device_info_event_handler(imu_data):
        print(f"Device Info: {imu_data}")

    def imu_event_handler(imu_data):
        print(f"IMU Data: {imu_data}")

    def attitude_event_handler(attitude_data):
        print(f"Attitude Data: {attitude_data}")

    def battery_event_handler(battery_data):
        print(f"Battery Data: {battery_data}")

    signal.signal(signal.SIGINT, signal_handler)

    fpv_interface = FpvInterface()
    # These callbacks are for use outside ROS2 land
    #fpv_interface.set_device_info_callback(device_info_event_handler)
    #fpv_interface.set_imu_callback(imu_event_handler)
    #fpv_interface.set_attitude_callback(attitude_event_handler)
    #fpv_interface.set_battery_callback(battery_event_handler)
    fpv_interface.start()


    #---- Send some channel data
    roll = 1500
    pitch = 1800
    yaw = 1500
    throttle = 1500
    arm = 1000
    mode = 1500
    turtle = 2000
    fpv_interface.set_channel_values(chT=throttle, chR=roll, chE=pitch, chA=yaw, aux1=arm, aux3=mode, aux4=turtle) # throttle, yaw, pitch, roll, arm, mode

    while runtime_exec:
        rclpy.spin_once(fpv_interface.ros2_node, timeout_sec=0.1)
        time.sleep(0.001)

    fpv_interface.stop()