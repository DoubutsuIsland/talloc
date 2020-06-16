import cv2 as cv
import numpy as np
from amas.agent import OBSERVER, Agent, NotWorkingError, Observer
from pino.ino import HIGH, LOW, Arduino


async def light(agent: Agent, ino: Arduino) -> None:
    await agent.fetch_from_observer()
    while agent.working():
        await agent.send_to("recorder", "is bright")
        _, mess = await agent.fetch_from_others()
        if mess:
            ino.digital_write(13, HIGH)
            continue
        ino.digital_write(13, LOW)
    return None


async def record(agent: Agent, cap: cv.VideoCapture) -> None:
    await agent.fetch_from_observer()
    widht = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    num_pixel = widht * height
    while agent.working():
        ret, frame = cap.read()
        if not ret:
            continue
        gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _sum = np.sum(gframe) / num_pixel
        if _sum >= 50:
            await agent.send_to("lighter", True)
        else:
            await agent.send_to("lighter", False)
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    await agent.send_to(OBSERVER, "session end")
    cv.destroyAllWindows()
    return None


async def kill(agent: Observer) -> None:
    await agent.send_all("start")
    while agent.working():
        _, mess = await agent.fetch_from_others()
        if mess == "session end":
            await agent.send_all(mess)
            agent.finish()
            break
    return None


async def quit(agent: Agent) -> None:
    while agent.working():
        _, mess = await agent.fetch_from_observer()
        if mess == "session end":
            agent.finish()
            break
    return None


async def sort(agent: Agent) -> None:
    while agent.working():
        try:
            await agent.sort_mail()
        except NotWorkingError:
            pass
    return None


if __name__ == '__main__':
    from amas.env import Environment
    from amas.connection import Register
    from pino.ino import Comport

    com = Comport() \
        .set_inofile("~/Github/GrudgeDuck/pino/example/proto.ino") \
        .set_baudrate(115200) \
        .set_port("/dev/ttyACM0") \
        .deploy() \
        .connect()

    ino = Arduino(com)

    cap = cv.VideoCapture(0)

    lighter = Agent("lighter") \
        .assign_task(light, ino=ino) \
        .assign_task(sort) \
        .assign_task(quit)
    recorder = Agent("recorder") \
        .assign_task(record, cap=cap) \
        .assign_task(sort) \
        .assign_task(quit)
    observer = Observer() \
        .assign_task(sort) \
        .assign_task(kill)

    rgist = Register([recorder, observer, lighter])
    env = Environment([recorder, observer, lighter])

    try:
        env.parallelize()
        env.join()
    finally:
        cap.release()
        pass
