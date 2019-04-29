from Queue import Queue


class IndexableQueue(Queue):
    """
    Queue that supports concurrent access to the data collection
    """

    def init_isFirst(self):
        self.isFirst = True

    def __getitem__(self, index):
        if index > self.qsize() or self.empty():
            return None
        with self.mutex:
            return self.queue[index]

    def __delitem__(self, index):
        with self.mutex:
            del self.queue[index]

    def get_n_frames(self, start_index, end_index):
        result = []
        for i in range(start_index, end_index):
            result.append(self.queue[i])
        return result

    def get_last_n_frame(self, n):
        result = []
        if n > self.qsize():
            n = self.qsize()
        for i in range(n):
            result.append(self.queue[i])
        return result

    def pull_last_n_frame(self, n):
        result = []
        if self.qsize() == 0:
            return None
        elif self.qsize() - n < 0:
            return None
        for i in range(n):
            result.append(self.queue[0])
            del self.queue[0]
        return result

    def empty_queue(self):
        with self.mutex:
            self.queue.clear()
            self.init_isFirst()

    def toList(self):
        return list(self.queue)