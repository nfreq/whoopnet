from whoopnet_io import WhoopnetIO
import logging
from inputs import devices, get_gamepad
import threading
import time
from queue import Queue, Empty
from enum import Enum
import random 
import math

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

class HandsetInputHandler(threading.Thread):
    RAW_MIN = -32062
    RAW_MAX = 32061
    OUTPUT_MIN = 1000  # Output for Position 1
    OUTPUT_MID = 1500  # Output for Position 2
    OUTPUT_MAX = 2000  # Output for Position 3

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

    def __init__(self, event_queue):
        super().__init__(daemon=True)  # Run as a daemon thread
        self.event_queue = event_queue
        self.running = True


    @staticmethod
    def scale_raw_to_position(value):
        """Scale raw units between RAW_MIN and RAW_MAX to 1000-2000."""
        if value <= HandsetInputHandler.RAW_MIN:
            return HandsetInputHandler.OUTPUT_MIN
        elif value >= HandsetInputHandler.RAW_MAX:
            return HandsetInputHandler.OUTPUT_MAX
        else:
            # Perform linear scaling between RAW_MIN (-32062) and RAW_MAX (32061) to (1000 - 2000)
            return int(
                HandsetInputHandler.OUTPUT_MIN + (value - HandsetInputHandler.RAW_MIN) * (HandsetInputHandler.OUTPUT_MAX - HandsetInputHandler.OUTPUT_MIN) / (HandsetInputHandler.RAW_MAX - HandsetInputHandler.RAW_MIN)
            )

    def run(self):
        gamepad = devices.gamepads
        if gamepad:
            logger.info(f"gamepad detected (ELRS RX): {gamepad[0].name}")
        else:
            logger.error("no handset/gamepad detected")
            self.running = False
        
        throttle = self.OUTPUT_MIN
        yaw = self.OUTPUT_MID
        pitch = self.OUTPUT_MID
        roll = self.OUTPUT_MID
        arm = self.OUTPUT_MIN
        auto = self.OUTPUT_MIN
        mode = self.OUTPUT_MAX
        turtle = self.OUTPUT_MIN
        while self.running:
            try:
                events = get_gamepad()
                for event in events:
                    control_action = self.CONTROL_MAPPING.get(event.code, None)
                    if control_action == ControlAction.HEARTBEAT:
                        pass
                        #logger.info("heartbeat")
                    elif control_action == ControlAction.ARM:
                        arm = 1000 if event.state == 0 else 2000
                    elif control_action == ControlAction.AUTO:
                        auto = 1000 if event.state == 0 else 2000
                    elif control_action == ControlAction.MODE:
                        mode = self.scale_raw_to_position(event.state)
                    elif control_action == ControlAction.TURTLE:
                        turtle = self.scale_raw_to_position(event.state)
                    elif control_action == ControlAction.THROTTLE:
                        throttle = self.scale_raw_to_position(event.state)
                    elif control_action == ControlAction.YAW:
                        yaw = self.scale_raw_to_position(event.state)
                    elif control_action == ControlAction.PITCH:
                        pitch = self.scale_raw_to_position(event.state)
                    elif control_action == ControlAction.ROLL:
                        roll = self.scale_raw_to_position(event.state)
                    else:
                        logger.debug(f"Unmapped Event - Code: {event.code} | State: {event.state}")
                    self.event_queue.put([arm, auto, mode, turtle, throttle, yaw, pitch, roll])
            except Exception as e:
                logger.error(f"Error in handset input thread: {e}")

    def stop(self):
        self.running = False


runtime_exec = True
def signal_handler(sig, frame):
    global runtime_exec
    print("\nSIGINT received and exiting")
    runtime_exec = False

if __name__ == "__main__":
    whoopnet_io = WhoopnetIO()
    whoopnet_io.start()
    whoopnet_io.init_rc_channels()

    handset_event_queue = Queue()
    handset_input_thread = HandsetInputHandler(event_queue=handset_event_queue)
    handset_input_thread.start()
        
    prev_arm = None
    prev_auto = 2000

    mix_active = False
    last_ai_action = time.time()

    sine_start = time.time()
    while runtime_exec:
        try:
            arm, auto, mode, turtle, throttle, yaw, pitch, roll = handset_event_queue.get_nowait()
            if prev_auto != auto: 
                prev_auto = auto
                if auto == 2000:
                    logger.info("manual control active")
                    mix_active = False
                elif auto == 1000:
                    logger.info("mixed ai control active")
                    mix_active = True
            
            if mix_active:
                whoopnet_io.set_rc_channels(aux1=arm, aux3=mode, aux8=auto)
            else:
                whoopnet_io.set_rc_channels(chT=throttle, chR=yaw, chA=roll, chE=pitch, aux1=arm, aux3=mode, aux4=turtle, aux8=auto)
            

            if prev_arm != arm: # this needs to come from vehicle
                prev_arm = arm
                if arm == 2000:
                    logger.info("armed")
                elif arm == 1000:
                    logger.info("disarmed")
               
        except Empty:
            pass

        if mix_active:
            if time.time() > last_ai_action + 0.02:
                last_ai_action = time.time()
                elapsed_time = time.time() - sine_start
                wave = math.sin(2 * math.pi * elapsed_time / 2)
                ai_action = int(1500 + wave * 400)
                whoopnet_io.set_rc_channels(ch=ai_action)
                

    handset_input_thread.stop()
    whoopnet_io.stop()