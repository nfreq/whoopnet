from whoopnet_io import WhoopnetIO
import logging
from inputs import devices, get_gamepad
from enum import Enum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler("whoopnet_manual_control.log")
console_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class ControlAction(Enum):
    PITCH = 'pitch'
    YAW = 'yaw'
    THROTTLE = 'throttle'
    ROLL = 'roll'
    ARM = 'arm'
    AUTO = 'auto'
    MODE = 'mode'
    TURTLE = 'turtle'
    HEARTBEAT = 'heartbeat'

def manual_control_handler(whoopnet_io):
    RAW_MIN = -32062
    RAW_MAX = 32061
    OUTPUT_MIN = 1000  # Output for Position 1
    OUTPUT_MID = 1500  # Output for Position 2
    OUTPUT_MAX = 2000  # Output for Position 3
    throttle = OUTPUT_MIN
    yaw = OUTPUT_MID
    pitch = OUTPUT_MID
    roll = OUTPUT_MID
    arm = OUTPUT_MIN
    auto = OUTPUT_MIN
    mode = OUTPUT_MIN
    turtle = OUTPUT_MIN

    CONTROL_MAPPING = {
        'ABS_RX': ControlAction.YAW,
        'ABS_Y': ControlAction.ROLL,
        'ABS_X': ControlAction.THROTTLE,
        'ABS_Z': ControlAction.PITCH,
        'BTN_C': ControlAction.ARM,
        'ABS_RZ': ControlAction.MODE,
        'BTN_SOUTH': ControlAction.AUTO,
        'SYN_REPORT': ControlAction.HEARTBEAT
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
                if control_action == ControlAction.HEARTBEAT:
                    pass
                    #logger.info("heartbeat")
                elif control_action == ControlAction.ARM:
                    arm = 1000 if event.state == 0 else 2000
                elif control_action == ControlAction.AUTO:
                    auto = 1000 if event.state == 0 else 2000
                elif control_action == ControlAction.MODE:
                    mode = scale_raw_to_position(event.state)
                elif control_action == ControlAction.TURTLE:
                    turtle = scale_raw_to_position(event.state)
                elif control_action == ControlAction.THROTTLE:
                    throttle = scale_raw_to_position(event.state)
                elif control_action == ControlAction.YAW:
                    yaw = scale_raw_to_position(event.state)
                elif control_action == ControlAction.PITCH:
                    pitch = scale_raw_to_position(event.state)
                elif control_action == ControlAction.ROLL:
                    roll = scale_raw_to_position(event.state)
                else:
                    logger.debug(f"Unmapped Event - Code: {event.code} | State: {event.state}")
                
            whoopnet_io.set_rc_channels(chT=throttle, chR=yaw, chA=roll, chE=pitch, aux1=arm, aux3=mode, aux4=turtle, aux8=auto)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    whoopnet_io = WhoopnetIO()
    whoopnet_io.start()
    whoopnet_io.init_rc_channels()
    manual_control_handler(whoopnet_io)
    whoopnet_io.stop()