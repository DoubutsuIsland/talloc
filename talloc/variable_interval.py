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

BLACK_BGR_MIN = np.array(expr_vars.get("black-bgr-min"))
BLACK_BGR_MAX = np.array(expr_vars.get("black-bgr-max"))
WHITE_BGR_MIN = np.array(expr_vars.get("white-bgr-min"))
WHITE_BGR_MAX = np.array(expr_vars.get("white-bgr-max"))
KERNEL = np.ones(tuple(expr_vars.get("kernel")), np.uint8)
COLOR_TEMP = expr_vars.get("color-temp")
ROI_WIDTH = 440
ROI_HEIGHT = 380
BLACK_ORIGIN = (117, 28)
WHITE_ORIGIN = (106, 45)
# KERNEL = np.ones((15, 15), np.uint8)
PMAX = ROI_WIDTH * ROI_HEIGHT

events: List[Tuple[float, int]] = []
trial = 0


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
                    intervals: List[float], blackout: float,
                    max_trial: int) -> None:
    global trial
    _ = await agent.recv_from_observer()
    print(f"{agent.addr} started")
    events.append((perf_counter(), 1))
    try:
        for interval in intervals:
            ino.digital_write(led, LIGHT_HIGH)
            events.append((perf_counter(), led))
            await agent.sleep(interval)
            agent.send_to(CAMAERA_ADDR, "is mouse in the box?")
            print(f"reward is set in {agent.addr}")
            while True:
                _, mess = await agent.recv()
                if mess:
                    break
            ino.digital_write(led, LIGHT_LOW)
            ino.digital_write(reward, HIGH)
            await agent.sleep(0.1)
            ino.digital_write(reward, LOW)
            events.append((perf_counter(), reward))
            trial += 1
            if trial >= max_trial:
                break
            await agent.sleep(blackout)
        events.append((perf_counter(), 0))
        if agent.working():
            agent.send_to(OBSERVER, "session terminated")
    except NotWorkingError:
        pass
    print(f"{agent.addr} stopped")
    ino.digital_write(led, LIGHT_LOW)
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


async def record(agent: Recorder, lcap: cv.VideoCapture,
                 rcap: cv.VideoCapture) -> None:
    black_roi = calc_roi(BLACK_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    white_roi = calc_roi(WHITE_ORIGIN, ROI_WIDTH, ROI_HEIGHT)
    now = datetime.datetime.now().strftime("%m%d%y%H%M%S")
    basename = "-".join([SUBJECT, CONDITION, now])
    fpath = join("data", basename)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    videoname = fpath + ".MP4"
    output = cv.VideoWriter(videoname, fourcc, 30.0, (1280, 480))
    _ = lcap.read()
    _ = rcap.read()
    agent.send_to(OBSERVER, "I'm ready")
    _ = await agent.recv_from_observer()
    print(f"{agent.addr} started")
    try:
        while agent.working():
            lret, lframe = lcap.read()
            lcap.set(cv.CAP_PROP_TEMPREATURE, COLOR_TEMP)
            rret, rframe = rcap.read()
            rcap.set(cv.CAP_PROP_TEMPREATURE, COLOR_TEMP)
            if not lret or not rret:
                continue
            lmouseish, lmframe = mouseish(lframe, black_roi, BLACK_BGR_MIN,
                                          BLACK_BGR_MAX)
            rmouseish, rmframe = mouseish(rframe, white_roi, WHITE_BGR_MIN,
                                          WHITE_BGR_MAX)
            if agent.lasked and lmouseish > BLACK_THRESHOLD:
                agent.send_to(BLACK_STIM_ADDR, True)
                agent.lasked = False
                print("reward is presented in left box")
            if agent.rasked and rmouseish > WHITE_THRESHOLD:
                agent.send_to(WHITE_STIM_ADDR, True)
                agent.rasked = False
                print("reward is presented in right box")
            stacked_frame = np.hstack((lframe, rframe))
            cv.imshow("black << --- >> white", stacked_frame)
            await agent.sleep(0.001)
            output.write(stacked_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    except NotWorkingError:
        pass
    cv.destroyAllWindows()
    if agent.working():
        agent.send_to(OBSERVER, "session terminated")
    print(f"{agent.addr} stopped")
    output.release()
    return None


async def asked(agent: Recorder) -> None:
    try:
        while agent.working():
            sender, mess = await agent.recv()
            if sender == BLACK_STIM_ADDR:
                agent.lasked = True
            elif sender == WHITE_STIM_ADDR:
                agent.rasked = True
    except NotWorkingError:
        pass
    return None


async def kill(agent: Observer, session_duration: float) -> None:
    _, mess = await agent.recv()
    agent.send_all("start")
    while agent.working():
        _ = await agent.try_recv(session_duration)
        agent.send_all("session terminated")
        agent.finish()
        break
    print(f"{agent.addr} stopped")
    return None


async def quit(agent: Agent) -> None:
    await agent.sleep(7.5)
    while agent.working():
        print(f"{agent.addr} await message from observer")
        _, mess = await agent.recv_from_observer()
        print(f"{agent.addr} recv {mess} from observer")
        if mess == "session end" or mess == "session terminated":
            agent.finish()
            break
    print(f"{agent.addr} quited")
    return None


if __name__ == '__main__':
    from amas.env import Environment
    from amas.connection import Register
    from pino.ino import OUTPUT, Comport
    from os.path import join

    BLACK_LED = expr_vars.get("black-led")
    BLACK_REWARD = expr_vars.get("black-reward")
    WHITE_LED = expr_vars.get("white-led")
    WHITE_REWARD = expr_vars.get("white-reward")
    BLACK_INTERVAL = expr_vars.get("black-interval")
    WHITE_INTERVAL = expr_vars.get("white-interval")
    BLACKOUT = expr_vars.get("blackout")
    SESSION_DURATION = expr_vars.get("session-duration")
    NUM_TRIAL = expr_vars.get("trial")
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

    now = datetime.datetime.now().strftime("%m%d%y")
    basename = "-".join([SUBJECT, CONDITION, now])
    fpath = join("data", basename)
    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # videoname = fpath + ".MP4"
    # output = cv.VideoWriter(videoname, fourcc, 30.0, (1280, 480))

    lstim = Agent(BLACK_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=BLACK_LED,
                     reward=BLACK_REWARD, intervals=black_intervals,
                     blackout=BLACKOUT, max_trial=NUM_TRIAL) \
        .assign_task(quit)

    rstim = Agent(WHITE_STIM_ADDR) \
        .assign_task(stimulate, ino=ino, led=WHITE_LED,
                     reward=WHITE_REWARD, intervals=white_intervals,
                     blackout=BLACKOUT, max_trial=NUM_TRIAL) \
        .assign_task(quit)

    lcap = cv.VideoCapture(LCAM)
    rcap = cv.VideoCapture(RCAM)

    recorder = Recorder(CAMAERA_ADDR) \
        .assign_task(record, lcap=lcap, rcap=rcap) \
        .assign_task(asked) \
        .assign_task(quit)
    observer = Observer().assign_task(kill, session_duration=SESSION_DURATION)

    rgist = Register([recorder, observer, lstim, rstim])
    env_rec = Environment([recorder])
    env_stim = Environment([lstim, rstim, observer])

    try:
        # env_stim.parallelize()
        # env_rec.run()
        # env_stim.join()
        env_rec.parallelize()
        env_stim.run()
        env_rec.join()
    finally:
        lcap.release()
        rcap.release()
        # output.release()
        fname = fpath + ".csv"
        with open(fname, "w") as f:
            f.write("time, event\n")
            for event in events:
                t, e = event
                f.write(f"{t}, {e}\n")
