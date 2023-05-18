import qi
import argparse
import sys
from maze import Maze
# export PYTHONPATH=~/programming/pynaoqi/lib/python2.7/site-packages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.0.197",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()

    # Marvin
    args.ip = "10.15.3.25"
    # Alan
    args.ip = "192.168.0.148"
    # Ada
    args.ip = "10.15.3.223"


    print("Connecting to ", args.ip)

    # for testing
    session = None

    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["Robot", "--qi-url=" + connection_url])

    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port))
        sys.exit(1)

    app.start()
    session = app.session

    maze = Maze(session)
    maze.keep_running()


if __name__ == "__main__":
    main()
