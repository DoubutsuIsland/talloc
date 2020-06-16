import datetime
from random import shuffle
from time import perf_counter
from typing import List, Tuple

import cv2 as cv
import numpy as np
from amas.agent import OBSERVER, Agent, NotWorkingError, Observer
from pino.config import Config
from pino.ino import HIGH, LOW, Arduino

ROI = Tuple[int, int, int, int]
BGR = List[int]

light_high = LOW
light_low = HIGH

LEFT_STIM_ADDR = "LSTIM"
RIGHT_STIM_ADDR = "RSTIM"
CAMAERA_ADDR = "CAM"

config = Config("hoge")
meta = config.get_metadata()
SUBJECT = meta.get("subject")
CONDITION = meta.get("condition")
expr_vars = config.get_experimental()
LEFT_THRESHOLD = expr_vars("left threshold")
RIGHT_THRESHOLD = expr_vars("right threshold")

# ROI_WIDTH = expr_vars.get("ROI width")
# ROI_HEIGHT = expr_vars.get("ROI height")
# LEFT_ORIGIN = expr_vars.get("left origin")
# RIGHT_ORIGIN = expr_vars.get("right origin")
# LEFT_BGR_MIN = expr_vars("left BGR min")
# LEFT_BGR_MAX = expr_vars("left BGR max")
# RIGHT_BGR_MIN = expr_vars("right BGR min")
# RIGHT_BGR_MAX = expr_vars("right BGR max")
# KERNEL = expr_vars("kernel")
ROI_WIDTH = 440
ROI_HEIGHT = 380
LEFT_ORIGIN = (117, 28)
RIGHT_ORIGIN = (106, 45)
LEFT_BGR_MIN = np.array([80, 0, 0])
LEFT_BGR_MAX = np.array([255, 255, 140])
RIGHT_BGR_MIN = np.array([0, 0, 0])
RIGHT_BGR_MAX = np.array([255, 255, 100])
KERNEL = np.ones((15, 15), np.uint8)


def init_table(interval: float, n: int) -> List:
    rate = 1 / interval
    table = []
    for i in range(n - 1):
        i += 1
        s = ((-np.log(1 - rate))**-1) * (1 + np.log(n) +
                                         (n - i) * np.log(n - i) -
                                         (n - i + 1) * np.log(n - i + 1))
        table.append(s)
    s = ((-np.log(1 - rate))**-1) * (1 + np.log(n) -
                                     (n - n + 1) * np.log(n - n + 1))
    table.append(s)
    shuffle(table)
    return table


async def stimulate(agent: Agent, ino: Arduino, led: int, reward: int,
                    intervals: List[float]) -> None:
    events: List[Tuple[float, str]] = []
    _ = await agent.fetch_from_observer()
    events.append((perf_counter(), "session start"))
    for interval in intervals:
        ino.digital_write(led, LOW)
        events.append((perf_counter(), "light on"))
        await agent.sleep(interval)
        agent.send_to(CAMAERA_ADDR, "is mouse in the box?")
        while True:
            _, mess = await agent.fetch_from_others()
            if mess:
                break
        ino.digital_write(led, light_low)
        ino.digital_write(reward, HIGH)
        events.append((perf_counter(), "reward on"))

    if not agent.working():
        agent.send_to(OBSERVER, "session terminated")

    # now = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    # fname = "-".join([SUBJECT, CONDITION, now]) + ".csv"
    # with open(fname, "w") as f:
    #     f.write("time, event\n")
    #     for event in events:
    #         t, e = event
    #         f.write(f"{t}, {e}\n")
    return None


def calc_roi(origin: Tuple[int, int], width: int,
             height: int) -> Tuple[int, int, int, int]:
    left, top = origin
    right = left + width
    bottom = top + height
    return (left, right, top, bottom)


def extract_roi(frame: np.ndarray, roi: tuple) -> np.ndarray:
    return frame[roi[2]:roi[3], roi[0]:roi[1]]


def mouseish(frame: np.ndarray, roi: ROI, bgr_min: BGR, bgr_max: BGR) -> float:
    gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    groi = extract_roi(gframe, roi)
    hroi = extract_roi(hframe, roi)
    hmask = cv.inRange(hroi, bgr_min, bgr_max)
    groi = (groi * hmask).astype(np.uint8)
    groi = cv.erode(groi, KERNEL, iterations=1)
    groi = cv.dilate(groi, KERNEL, iterations=1)
    return np.sum(groi)


class Recorder(Agent):
    def __init__(self, addr: str):
        super().__init__(addr)
        self.lasked = False
        self.rasked = False


async def record(agent: Recorder, lcap: cv.VideoCapture, rcap: cv.VideoCapture,
                 output: cv.VideoWriter) -> None:
    left_roi = calc_roi(LEFT_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    right_roi = calc_roi(RIGHT_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    _ = await agent.fetch_from_observer()
    while agent.working():
        lret, lframe = lcap.read()
        rret, rframe = rcap.read()
        if lret and rret:
            continue
        lmouseish = mouseish(lframe, left_roi, LEFT_BGR_MIN, LEFT_BGR_MAX)
        rmouseish = mouseish(rframe, right_roi, RIGHT_BGR_MIN, RIGHT_BGR_MAX)
        if agent.lasked and lmouseish > LEFT_THRESHOLD:
            await agent.send_to(LEFT_STIM_ADDR, "mouse is in left side")
            agent.lasked = False
        if agent.rasked and rmouseish > RIGHT_THRESHOLD:
            await agent.send_to(RIGHT_STIM_ADDR, "mouse is in right side")
            agent.rasked = False
        cv.imshow("Left << --- >> Right", np.hstack((lframe, rframe)))
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    if agent.working():
        agent.send_to(OBSERVER, "session terminated")
    return None


async def asked(agent: Recorder) -> None:
    while agent.working():
        sender, _ = await agent.fetch_from_others()
        if sender == LEFT_STIM_ADDR:
            agent.lasked = True
        elif sender == RIGHT_STIM_ADDR:
            agent.rasked = True
    return None


async def sort(agent: Agent) -> None:
    while agent.working():
        try:
            await agent.sort_mail()
        except NotWorkingError:
            pass
    return None


async def kill(agent: Observer) -> None:
    while agent.working():
        _, mess = await agent.fetch_from_others()
        if mess == "session end" or mess == "session terminated":
            await agent.send_all(mess)
            agent.finish()
            break
    return None


async def quit(agent: Agent) -> None:
    while agent.working():
        _, mess = await agent.fetch_from_observer()
        if mess == "session end" or mess == "session terminated":
            agent.finish()
            break
    return None


if __name__ == '__main__':
    from amas.env import Environment
    from amas.connection import Register
    from pino.ino import OUTPUT, Comport

    LEFT_LED = expr_vars.get("left-led")
    LEFT_REWARD = expr_vars.get("left-reward")
    RIGHT_LED = expr_vars.get("right-led")
    RIGHT_REWARD = expr_vars.get("right-reward")
    LEFT_INTERVAL = expr_vars.get("left-interval")
    RIGHT_INTERVAL = expr_vars.get("right-interval")
    SESSION_DURATION = expr_vars("session-duration")
    LCAM = expr_vars.get("left-cam")
    RCAM = expr_vars.get("right-cam")

    com = Comport().apply_settings(config.get_comport()).deploy().connect()
    ino = Arduino(com)

    ino.set_pinmode(LEFT_LED, OUTPUT)
    ino.set_pinmode(RIGHT_LED, OUTPUT)
    ino.set_pinmode(LEFT_REWARD, OUTPUT)
    ino.set_pinmode(RIGHT_REWARD, OUTPUT)

    leftn = SESSION_DURATION // LEFT_INTERVAL
    left_intervals = init_table(LEFT_INTERVAL, leftn)
    rightn = SESSION_DURATION // RIGHT_INTERVAL
    right_intervals = init_table(LEFT_INTERVAL, rightn)

    now = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    basename = "-".join([SUBJECT, CONDITION, now])
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    videoname = basename + ".MP4"
    output = cv.VideoWriter(videoname, fourcc, 30.0, (640, 480))

    lstim = Agent(LEFT_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=LEFT_LED,
                     reward=LEFT_REWARD, intervals=left_intervals) \
        .assign_task(sort) \
        .assign_task(quit)

    rstim = Agent(RIGHT_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=RIGHT_LED,
                     reward=RIGHT_REWARD, intervals=right_intervals) \
        .assign_task(sort) \
        .assign_task(quit)

    lcap = cv.VideoCapture(LCAM)
    rcap = cv.VideoCapture(RCAM)

    recorder = Recorder(CAMAERA_ADDR) \
        .assign_task(record, lcap=lcap, rcap=rcap, output=output) \
        .assign_task(sort) \
        .assign_task(quit)
    observer = Observer().assign_task(kill)

    rgist = Register([recorder, observer])
    env = Environment([recorder, observer])

    try:
        env.run()
    finally:
        lcap.release()
        rcap.release()
        output.release()
