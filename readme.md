webrtc test with aiortc

# How to use:
1. [Bring up the server (see below)](#server)
2. locate it's IP. Lets assume xxx.xxx.xxx.xxx
3. Ensure the computer running the browse can hit the server. Try hitting `http://xxx.xxx.xxx.xxx:9999`
5. and then visit https://party.photoboo.app/. Final url should be something like https://party.photoboo.app/?http://xxx.xxx.xxx.xxx:9999


# Components

## Server

```sh
$ cd server
$ pip install -r requirements.txt
$ python app.py
```

## Frontend
This is the web client, based on Svelte + Vite + Tailwind.

```sh
$ cd website
$ npm i
$ npm run dev
```

## Caddy
Proxies everything. Does SSL termination

```
$ cd server
$ sudo caddy run
```
