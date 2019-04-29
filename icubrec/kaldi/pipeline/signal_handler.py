import signal


class SignalHandler:
    silent = False
    should_stop = False

    @staticmethod
    def signal_handler(signal, frame):
        SignalHandler.should_stop = True
        if not SignalHandler.silent:
            print("Termination signal received.")


signal.signal(signal.SIGINT, SignalHandler.signal_handler)
signal.signal(signal.SIGTERM, SignalHandler.signal_handler)
