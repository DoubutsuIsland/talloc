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

LIGHT_HIGH = LOW
LIGHT_LOW = HIGH

BLACK_STIM_ADDR = "LSTIM"
WHITE_STIM_ADDR = "RSTIM"
CAMAERA_ADDR = "CAM"

config = Config("./config/variable_interval.yml")
meta = config.get_metadata()
SUBJECT = meta.get("subject")
CONDITION = meta.get("condition")
expr_vars = config.get_experimental()
BLACK_THRESHOLD = expr_vars.get("black-threshold")
WHITE_THRESHOLD = expr_vars.get("white-threshold")

# ROI_WIDTH = expr_vars.get("ROI width")
# ROI_HEIGHT = expr_vars.get("ROI height")
# BLACK_ORIGIN = expr_vars.get("black origin")
# WHITE_ORIGIN = expr_vars.get("white origin")
BLACK_BGR_MIN = np.array(expr_vars.get("black-bgr-min"))
BLACK_BGR_MAX = np.array(expr_vars.get("black-bgr-max"))
WHITE_BGR_MIN = np.array(expr_vars.get("white-bgr-min"))
WHITE_BGR_MAX = np.array(expr_vars.get("white-bgr-max"))
# KERNEL = expr_vars("kernel")
ROI_WIDTH = 440
ROI_HEIGHT = 380
BLACK_ORIGIN = (117, 28)
WHITE_ORIGIN = (106, 45)
# BLACK_BGR_MIN = np.array([30, 0, 0])
# BLACK_BGR_MAX = np.array([255, 255, 100])
# WHITE_BGR_MIN = np.array([80, 0, 0])
# WHITE_BGR_MAX = np.array([255, 255, 140])
KERNEL = np.ones((15, 15), np.uint8)
PMAX = ROI_WIDTH * ROI_HEIGHT


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
        # print(f"{agent.addr} {interval}")
        ino.digital_write(led, LIGHT_HIGH)
        events.append((perf_counter(), "light on"))
        await agent.sleep(interval)
        await agent.send_to(CAMAERA_ADDR, "is mouse in the box?")
        while True:
            _, mess = await agent.fetch_from_others()
            if mess:
                break
        ino.digital_write(led, LIGHT_LOW)
        ino.digital_write(reward, HIGH)
        await agent.sleep(0.1)
        ino.digital_write(reward, LOW)
        events.append((perf_counter(), "reward on"))
        await agent.sleep(5)

    if agent.working():
        await agent.send_to(OBSERVER, "session terminated")

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


def mouseish(frame: np.ndarray, roi: ROI, bgr_min: BGR,
             bgr_max: BGR) -> Tuple[float, np.ndarray]:
    gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hframe = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    groi = extract_roi(gframe, roi)
    hroi = extract_roi(hframe, roi)
    hmask = cv.inRange(hroi, bgr_min, bgr_max)
    groi = (groi * hmask).astype(np.uint8)
    groi = cv.erode(groi, KERNEL, iterations=1)
    groi = cv.dilate(groi, KERNEL, iterations=1)
    return np.sum(groi) / PMAX, groi


class Recorder(Agent):
    def __init__(self, addr: str):
        super().__init__(addr)
        self.lasked = False
        self.rasked = False


async def record(agent: Recorder, lcap: cv.VideoCapture, rcap: cv.VideoCapture,
                 output: cv.VideoWriter) -> None:
    black_roi = calc_roi(BLACK_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    white_roi = calc_roi(WHITE_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    await agent.send_to(OBSERVER, "I'm ready")
    _ = await agent.fetch_from_observer()
    while agent.working():
        lret, lframe = await agent.call_async(lcap.read)
        rret, rframe = await agent.call_async(rcap.read)
        if not lret and not rret:
            continue
        lmouseish, lmframe = mouseish(lframe, black_roi, BLACK_BGR_MIN,
                                      BLACK_BGR_MAX)
        rmouseish, rmframe = mouseish(rframe, white_roi, WHITE_BGR_MIN,
                                      WHITE_BGR_MAX)
        print(f"black: {lmouseish}")
        print(f"white: {rmouseish}")
        if agent.lasked and lmouseish > BLACK_THRESHOLD:
            await agent.send_to(BLACK_STIM_ADDR, True)
            agent.lasked = False
        if agent.rasked and rmouseish > WHITE_THRESHOLD:
            await agent.send_to(WHITE_STIM_ADDR, True)
            agent.rasked = False
        stacked_frame = np.hstack((lframe, rframe))
        cv.imshow("black << --- >> white", stacked_frame)
        output.write(stacked_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()
    if agent.working():
        await agent.send_to(OBSERVER, "session terminated")
    return None


async def asked(agent: Recorder) -> None:
    while agent.working():
        sender, mess = await agent.fetch_from_others()
        print(mess)
        if sender == BLACK_STIM_ADDR:
            agent.lasked = True
        elif sender == WHITE_STIM_ADDR:
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
    _, mess = await agent.fetch_from_others()
    await agent.send_all("start")
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

    BLACK_LED = expr_vars.get("black-led")
    BLACK_REWARD = expr_vars.get("black-reward")
    WHITE_LED = expr_vars.get("white-led")
    WHITE_REWARD = expr_vars.get("white-reward")
    BLACK_INTERVAL = expr_vars.get("black-interval")
    WHITE_INTERVAL = expr_vars.get("white-interval")
    SESSION_DURATION = expr_vars.get("session-duration")
    LCAM = expr_vars.get("black-cam")
    RCAM = expr_vars.get("white-cam")

    com = Comport().apply_settings(config.get_comport()).deploy().connect()
    ino = Arduino(com)

    ino.set_pinmode(BLACK_LED, OUTPUT)
    ino.set_pinmode(WHITE_LED, OUTPUT)
    ino.set_pinmode(BLACK_REWARD, OUTPUT)
    ino.set_pinmode(WHITE_REWARD, OUTPUT)

    blackn = SESSION_DURATION // BLACK_INTERVAL
    black_intervals = init_table(BLACK_INTERVAL, blackn)
    whiten = SESSION_DURATION // WHITE_INTERVAL
    white_intervals = init_table(BLACK_INTERVAL, whiten)

    now = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    basename = "-".join([SUBJECT, CONDITION, now])
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    videoname = basename + ".MP4"
    output = cv.VideoWriter(videoname, fourcc, 30.0, (1280, 480))

    lstim = Agent(BLACK_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=BLACK_LED,
                     reward=BLACK_REWARD, intervals=black_intervals) \
        .assign_task(sort) \
        .assign_task(quit)

    rstim = Agent(WHITE_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=WHITE_LED,
                     reward=WHITE_REWARD, intervals=white_intervals) \
        .assign_task(sort) \
        .assign_task(quit)

    lcap = cv.VideoCapture(LCAM)
    rcap = cv.VideoCapture(RCAM)

    recorder = Recorder(CAMAERA_ADDR) \
        .assign_task(record, lcap=lcap, rcap=rcap, output=output) \
        .assign_task(asked) \
        .assign_task(sort) \
        .assign_task(quit)
    observer = Observer().assign_task(kill).assign_task(sort)

    rgist = Register([recorder, observer, lstim, rstim])
    env_rec = Environment([recorder, observer])
    env_stim = Environment([lstim, rstim])

    try:
        env_stim.parallelize()
        env_rec.run()
        env_stim.join()
        # env_rec.parallelize()
        # env_stim.run()
        # env_rec.join()
    finally:
        lcap.release()
        rcap.release()
        output.release()
