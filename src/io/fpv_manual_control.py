from fpv_interface import FpvInterface
import time
import logging
from inputs import devices, get_gamepad

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler("fpv_passthrough.log")
console_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

OUTPUT_MIN = 1000  # Output for Position 1
OUTPUT_MID = 1500  # Output for Position 2
OUTPUT_MAX = 2000  # Output for Position 3

throttle = OUTPUT_MIN
yaw = OUTPUT_MID
pitch = OUTPUT_MID
roll = OUTPUT_MID
arm = OUTPUT_MIN
mode = OUTPUT_MIN
turtle = OUTPUT_MIN

def manual_control_handler(fpv_interface):
    global throttle, yaw, pitch, roll, arm, mode, turtle
    RAW_MIN = -32062
    RAW_MAX = 32061

    CONTROL_MAPPING = {
        'ABS_RX': 'pitch',
        'ABS_Y': 'yaw',
        'ABS_X': 'throttle',
        'ABS_Z': 'roll',
        'BTN_C': 'arm',
        'ABS_RZ': 'mode',
        'BTN_SOUTH': 'turtle'
    }

    def scale_raw_to_position(value):
        """Scale raw units between RAW_MIN and RAW_MAX to 1000-2000."""
        if value <= RAW_MIN:
            return OUTPUT_MIN
        elif value >= RAW_MAX:
            return OUTPUT_MAX
        else:
            # Perform linear scaling between RAW_MIN (-32062) and RAW_MAX (32061) to (1000 - 2000)
            return int(OUTPUT_MIN + (value - RAW_MIN) * (OUTPUT_MAX - OUTPUT_MIN) / (RAW_MAX - RAW_MIN))
        
    gamepad = devices.gamepads
    if not gamepad:
        logger.error("No gamepad connected!")
        return
    
    logger.info(f"gamepad detected (ELRS RX): {gamepad[0].name}")
    try:
        while True:
            events = get_gamepad()
            for event in events:
                control_action = CONTROL_MAPPING.get(event.code, None)
                if control_action == 'arm':
                    arm = 1000 if event.state == 0 else 2000
                elif control_action == 'mode':
                    mode = scale_raw_to_position(event.state)
                elif control_action == 'turtle':
                    turtle = scale_raw_to_position(event.state)
                elif control_action == 'throttle':
                    throttle = scale_raw_to_position(event.state)
                elif control_action == 'yaw':
                    yaw = scale_raw_to_position(event.state)
                elif control_action == 'pitch':
                    pitch = scale_raw_to_position(event.state)
                elif control_action == 'roll':
                    roll = scale_raw_to_position(event.state)
                else:
                    logger.debug(f"Unmapped Event - Code: {event.code} | State: {event.state}")
                
            fpv_interface.set_channel_values(chT=throttle, chR=yaw, chE=roll, chA=pitch, aux1=arm, aux2=arm, aux3=mode, aux4=turtle)
    except KeyboardInterrupt:
        pass


def main():
    logger.info("FPV Manual Control")

    fpv_interface = FpvInterface()
    fpv_interface.start()
    time.sleep(1)
    
    fpv_interface.set_channel_values(chT=throttle, chR=yaw, chE=roll, chA=pitch, aux1=arm, aux2=arm, aux3=mode, aux4=turtle)        #initialize values

    manual_control_handler(fpv_interface)

    fpv_interface.stop()

if __name__ == "__main__":
    main()