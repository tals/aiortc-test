from datetime import datetime

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import *

import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from av import VideoFrame
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from PIL import Image
from pydantic import BaseModel


PY_ROOT = Path(__file__).parent
app = FastAPI(openapi_url=None)
# app = FastAPI()
SERVE_WEBSITE = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()
relay = MediaRelay()

image_transform = None


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform=None):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform or "cartoon"

    async def recv(self):
        frame = await self.track.recv()
        if image_transform:
            tensor = frame.to_ndarray(format="rgb24")
            tensor = image_transform(tensor)

            # put it back together
            if tensor is None:
                return frame
            try:
                new_frame = VideoFrame.from_ndarray(tensor, format="rgb24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                return new_frame
            except Exception as e:
                logger.exception("Something bad happened")
                print("Something bad happened", e, tensor.shape)

        else:
            return self.base_transform(frame)
        # await asyncio.sleep(0.1)

    def base_transform(self, frame):
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame

if SERVE_WEBSITE:
    WEBSITE_ROOT = Path("../website/dist")
    app.mount("/assets", StaticFiles(directory=WEBSITE_ROOT / "assets"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(WEBSITE_ROOT / "index.html")


class Offer(BaseModel):
    sdp: Any
    type: Any


recorder = None
@app.post("/offer")
async def offer(offer: Offer, request: Request):
    global recorder
    print("offer recieved")
    offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s" % request.client.host)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global transform_track
        log_info(f"Track {track.kind} received")

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            recorder.addTrack(relay.subscribe(track))
            transform_track = VideoTransformTrack(
                relay.subscribe(track), transform=None
            )
            pc.addTrack(transform_track)

        @track.on("ended")
        async def on_ended():
            log_info(f"Track {track.kind} ended")
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.on_event("shutdown")
async def on_shutdown():
    print("Preparing to shutdown.")
    # close peer connections
    coros = [pc.close() for pc in pcs]
    print("Preparing to shutdown: waiting on PCs")
    await asyncio.gather(*coros)
    pcs.clear()
    print("Shut down!")


def main(raw_args=[]):
    import uvicorn

    global args
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9999, help="Port for HTTP server (default: 9999)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args(raw_args)
    uvicorn.run(app, host=args.host, port=args.port, loop="asyncio")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


class Timer:
    _start: datetime
    def __init__(self):
        self.samples = []
        self._start = datetime.now()

    def lap(self):
        now = datetime.now()
        self.samples.append((now - self._start).total_seconds())
        self._start = now

    def on_start(self):
        self._start = datetime.now()

    def print_stats(self):
        if not self.samples:
            return
            
        samples = np.array(self.samples) * 1000
        print(f"median {np.median(samples):n} min: {np.min(samples):n} max: {np.max(samples):n}")

    def reset(self):
        self.samples = []
        self._start = datetime.now()

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
