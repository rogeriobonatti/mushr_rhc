from queue import Queue

q = Queue(maxsize=5)
q.put(1)
q.put(2)
q.put(3)

for i in q.queue:
    print(i)