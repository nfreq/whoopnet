import logging
from inputs import devices, get_gamepad
import threading
import time
from queue import Queue, Empty
from enum import Enum
import random 
import math
from whoopnet_io import WhoopnetIO

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

class RCChannelMixer:
    def __init__(self):
        self.manual_channels = {
            'chT': 1000,  # Default throttle
            'chR': 1500,  # Default yaw
            'chA': 1500,  # Default roll
            'chE': 1500,  # Default pitch
            'aux1': 2000, # Default arm
            'aux3': 1500, # Default mode
            'aux4': 1500, # Default turtle
            'aux8': 1500, # Default auto
        }
        self.ai_channels = {key: None for key in ('chT', 'chR', 'chA', 'chE')}  # AI cannot control aux channels

    def update_manual(self, **kwargs):
        self.manual_channels.update(kwargs)

    def update_ai(self, **kwargs):
        self.ai_channels.update(kwargs)

    def mix_channels(self, mix_active):
        """Mix manual and AI channels based on the mix_active flag."""
        return {
            key: (
                self.ai_channels[key] if mix_active and key in self.ai_channels and self.ai_channels[key] is not None
                else self.manual_channels[key]
            )
            for key in self.manual_channels
        }


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


class RCMixer(threading.Thread):
    def __init__(self, whoopnet_io):
        super().__init__(daemon=True)
        self.whoopnet_io = whoopnet_io
        self.handset_event_queue = Queue()
        self.handset_input_thread = HandsetInputHandler(event_queue=self.handset_event_queue)
        self.mixer = RCChannelMixer()

        self.prev_arm = None
        self.prev_auto = 2000
        self.mix_active = False
        self.last_ai_action = time.time()
        self.sine_start = time.time()
        self.runtime_exec = True

    def start_mixer(self):
        self.handset_input_thread.start()
        self.start()

    def stop_mixer(self):
        self.runtime_exec = False
        self.handset_input_thread.stop()

    def run(self):
        while self.runtime_exec:
            try:
                arm, auto, mode, turtle, throttle, yaw, pitch, roll = self.handset_event_queue.get_nowait()

                if self.prev_auto != auto:
                    self.prev_auto = auto
                    if auto == 2000:
                        logger.info("manual control active")
                        self.mix_active = False
                    elif auto == 1000:
                        logger.info("mixed AI control active")
                        self.mix_active = True

                self.mixer.update_manual(chT=throttle, chR=yaw, chA=roll, chE=pitch, aux1=arm, aux3=mode, aux8=auto)

                if self.prev_arm != arm:
                    self.prev_arm = arm
                    if arm == 2000:
                        logger.info("armed")
                    elif arm == 1000:
                        logger.info("disarmed")

            except Empty:
                pass

            if self.mix_active:
                if time.time() > self.last_ai_action + 0.02:
                    self.last_ai_action = time.time()
                    elapsed_time = time.time() - self.sine_start
                    wave1 = math.sin(2 * math.pi * elapsed_time / 3)
                    ai_action1 = int(1500 + wave1 * 200)

                    wave2 = math.sin(2 * math.pi * elapsed_time / 0.75)
                    ai_action2 = int(1500 + wave2 * 100)
                    self.mixer.update_ai(chT=ai_action1, chR=ai_action2)


            mixed_channels = self.mixer.mix_channels(self.mix_active)
            self.whoopnet_io.set_rc_channels(**mixed_channels)


runtime_exec = True
def signal_handler(sig, frame):
    global runtime_exec
    print("\nSIGINT received and exiting")
    runtime_exec = False

if __name__ == "__main__":
    from whoopnet_io import WhoopnetIO

    whoopnet_io = WhoopnetIO()
    whoopnet_io.start()
    whoopnet_io.init_rc_channels()

    mixer = RCMixer(whoopnet_io)
    try:
        mixer.start_mixer()
        mixer.join()
    finally:
        mixer.stop_mixer()
